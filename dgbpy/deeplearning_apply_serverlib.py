#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : A. Huck
# DATE     : June 2019
#
# Deep learning apply server
#
#

import io
import json
import numpy as np
import os
import psutil
import selectors
import struct
import sys
import traceback as tb

from odpy.common import *
import dgbpy.keystr as dgbkeys
from dgbpy import hdf5 as dgbhdf5
from dgbpy import mlio as dgbmlio
from dgbpy import mlapply as dgbmlapply

class ParentExitCommand(Exception):
    pass
class ExitCommand(Exception):
    pass

class ModelApplier:
    def __init__(self, modelfnm):
        self.pars_ = None
        self.fakeapply_ = False
        self.scaler_ = None
        self.img2img_ = None
        self.is1dmodel_ = None
        self.is2dmodel_ = None
        self.is3dmodel_ = None
        self.isflat_inlinemodel_ = False
        self.isflat_xlinemodel_ = False
        self.datais2d_ = False
        self.info_ = self._get_info(modelfnm)
        self.needtranspose_ = False
        self.needztranspose_ = False
        self._set_transpose()
        self.model_ = None
        self.applyinfo_ = None
        self.batchsize_ = None
        self.applydir_ = None
        self.debugstr_ = ''

    def _get_info(self, modelfnm):
        info = dgbmlio.getInfo( modelfnm, True )
        self.img2img_ = dgbhdf5.isImg2Img(info)
        self.is1dmodel_ = dgbhdf5.is1DModel( info )
        self.is2dmodel_ = dgbhdf5.is2DModel( info )
        self.is3dmodel_ = dgbhdf5.is3DModel( info )
        inpshape = info[dgbkeys.inpshapedictstr]
        if not self.datais2d_ and self.is2dmodel_:
          self.isflat_inlinemodel_ = inpshape[0] == 1 and inpshape[1] > 1
          self.isflat_xlinemodel_ = inpshape[0] > 1 and inpshape[1] == 1
          if self.isflat_inlinemodel_:
            self.applydir_ = dgbkeys.inlinestr
          elif self.isflat_xlinemodel_:
            self.applydir_ = dgbkeys.crosslinestr

        return info
    
    def setParameters(self, pars):
        if 'fake_apply' in pars and pars['fake_apply']:
            self.fakeapply_ = pars['fake_apply']
            self.info_[dgbkeys.plfdictstr] = dgbkeys.numpyvalstr

        if 'data_is2d' in pars:
            self.datais2d_ = pars['data_is2d']

        if self.fakeapply_:
            self.applyinfo_ = dgbmlio.getApplyInfo( self.info_ )
        else:
            self.applyinfo_ = dgbmlio.getApplyInfo( self.info_, pars )

        if dgbhdf5.applyGlobalStd( self.info_ ):
          self.scaler_ = self.getScaler( pars )

        if self.info_[dgbkeys.plfdictstr] == dgbkeys.kerasplfnm:
            from dgbpy import dgbkeras
            if dgbkeys.prefercpustr in pars:
                dgbkeras.set_compute_device( pars[dgbkeys.prefercpustr] )
            if dgbkeras.defbatchstr in pars:
                self.batchsize_ = pars[dgbkeras.defbatchstr]
        elif self.info_[dgbkeys.plfdictstr] == dgbkeys.torchplfnm:
            from dgbpy import dgbtorch
            if dgbkeys.prefercpustr in pars:
                dgbtorch.set_compute_device( pars[dgbkeys.prefercpustr] )
            if dgbtorch.defbatchstr in pars:
                self.batchsize_ = pars[dgbtorch.defbatchstr]

        if not self.datais2d_ and self.is2dmodel_:
            if self.isflat_inlinemodel_:
              self.applydir_ = dgbkeys.inlinestr
            elif self.isflat_xlinemodel_:
              self.applydir_ = dgbkeys.crosslinestr

            if 'apply_dir' in pars:
              dir = pars['apply_dir']
              if dir==dgbkeys.inlinestr or dir==dgbkeys.crosslinestr or dir==dgbkeys.averagestr or \
                 dir==dgbkeys.minstr or dir==dgbkeys.maxstr:
                   self.applydir_ = dir

        if self.fakeapply_:
            return

        modelfnm = self.info_[dgbkeys.filedictstr]
        (self.model_,self.info_) = dgbmlio.getModel( modelfnm, fortrain=False )
        if 'infer_size' in pars:
          self.info_[dgbkeys.inpshapedictstr] = pars['infer_size']
          if self.img2img_:
            self.info_[dgbkeys.outshapedictstr] = pars['infer_size']

        self._set_transpose()

    def _get_swapaxes_dim(self, arr):
        ndim = len(arr.shape)
        if ndim == 5:
            return 2, 3
        return ndim-3, ndim-2

    def _set_transpose(self):
        if dgbhdf5.isSeisClass( self.info_ ) or \
            dgbhdf5.isImg2Img( self.info_ ):
            self.needtranspose_ = dgbhdf5.applyArrTranspose( self.info_ )
            self.needztranspose_ = dgbhdf5.applyArrZTranspose( self.info_ )
        else:
            self.needtranspose_ = False
            self.needztranspose_ = False
    
    def _usePar(self, pars):
        self.pars_ = pars

    def hasModel(self):
        return self.model_ != None

    def getScaler( self, outputs ):
        if not 'scales' in outputs:
            return self.scaler_

        scales = outputs['scales']
        means = list()
        stddevs = list()
        scaleratios = list()
        for scl in scales:
            means.append( scl['avg'] )
            stddevs.append( scl['stdev'] )
            scaleratios.append( scl['scaleratio'] )

        if len(means) > 0:
            from dgbpy import dgbscikit
            self.scaler_ = dgbscikit.getNewScaler( means, stddevs )
        inputs = self.info_[dgbkeys.inputdictstr]
        if dgbhdf5.isLogInput( self.info_ ):
            inputs = self.info_[dgbkeys.inputdictstr]
            firstinpnm = next(iter(inputs))
            if firstinpnm in inputs:
                inp = inputs[firstinpnm]
                if dgbkeys.scaledictstr in inp:
                    self.scaler_ = inp[dgbkeys.scaledictstr]
        elif outputs[dgbkeys.surveydictstr] in inputs:
            survdirnm = outputs[dgbkeys.surveydictstr]
            inp = inputs[survdirnm]
            if dgbkeys.scaledictstr in inp:
                inpscale = inp[dgbkeys.scaledictstr]
                if dgbkeys.collectdictstr in inp:
                  attribs = inp[dgbkeys.collectdictstr]
                  for scl in scales:
                      applykey = scl[dgbkeys.dbkeydictstr]
                      applynm = scl[dgbkeys.namedictstr]
                      iattr = 0
                      for attribnm in attribs:
                          attrib = attribs[attribnm]
                          if attrib[dgbkeys.dbkeydictstr] == applykey or \
                             attribnm == applynm:
                             idx = attrib[dgbkeys.iddictstr]
                             self.scaler_.scale_[idx] = inpscale.scale_[idx]
                             self.scaler_.mean_[idx] = inpscale.mean_[idx]
                             break
                          else:
                            self.scaler_.scale_[iattr] *= scaleratios[iattr]
                          iattr += 1
                else:
                  for i in range(len(inpscale.scale_)):
                    means.append( inpscale.mean_[i] )
                    stddevs.append( inpscale.scale_[i] )
                  if len(means) > 0:
                    from dgbpy import dgbscikit
                    self.scaler_ = dgbscikit.getNewScaler( means, stddevs )

        return self.scaler_

    def preprocess(self, samples, swapaxes):
        if swapaxes:
            samples = samples.swapaxes(*self._get_swapaxes_dim(samples))

        if dgbhdf5.applyLocalStd( self.info_ ):
            from dgbpy import dgbscikit
            self.scaler_ = dgbscikit.getScaler( samples, True )
        elif dgbhdf5.applyNormalization( self.info_ ):
            from dgbpy import dgbscikit
            self.scaler_ = dgbscikit.getNewMinMaxScaler( samples )
        elif dgbhdf5.applyMinMaxScaling( self.info_ ):
            from dgbpy import dgbscikit
            self.scaler_ = dgbscikit.getNewMinMaxScaler( samples, maxout=255 )
        elif dgbhdf5.applyRangeScaling( self.info_ ):
            from dgbpy import dgbscikit
            self.scaler_ = dgbscikit.getNewRangeScaler( samples )
        elif dgbhdf5.applyGlobalStd( self.info_ ):
            if self.scaler_ == None:
                self.debug_msg( 'Missing scaler for global standardization' )
                raise TypeError

        if self.scaler_ != None:
            from dgbpy import dgbscikit
            samples = dgbscikit.scale( samples, self.scaler_ )

        if self.needtranspose_:
            samples = np.transpose( samples, axes=(0,1,4,3,2) )
        elif self.needztranspose_:
            samples = np.transpose( samples, axes=(0,1,4,2,3) )
        return samples

    def postprocess(self, samples, swapaxes):
        if self.needtranspose_:
            samples = np.transpose( samples, axes=(0,1,4,3,2) )
        elif self.needztranspose_:
            samples = np.transpose( samples, axes=(0,1,3,4,2) )

        if self.datais2d_ and len(samples.shape)==5:
            samples = samples[:,:,samples.shape[2]//2,:,:]
        elif swapaxes:
            samples = samples.swapaxes(*self._get_swapaxes_dim(samples))

        if dgbhdf5.unscaleOutput( self.info_ ):
            if self.scaler_:
                from dgbpy import dgbscikit
                samples = dgbscikit.unscale( samples, self.scaler_ )

        return samples

    def flatModelApply(self, inp, samples, samples_shape):
        inpshape = self.info_[dgbkeys.inpshapedictstr]
        nrout = dgbhdf5.getNrOutputs(self.info_)
        outshape = (1, nrout, *inp.shape[1:])
        outdata = np.zeros(outshape, dtype=inp.dtype)
        applydata = np.empty(samples_shape, dtype=inp.dtype)
        for idx in range(max(inpshape[0], inpshape[1])):
            applydata[:,:,0] = samples[:,:,idx]
            ret = dgbmlapply.doApply( self.model_, self.info_, applydata, \
                                    scaler=None, applyinfo=self.applyinfo_, \
                                    batchsize=self.batchsize_ )
            if dgbkeys.preddictstr in ret:
                outdata[0,:,idx] = ret[dgbkeys.preddictstr][:,0]

        return outdata

    def doWork(self, inp):
        nrattribs = dgbhdf5.getNrAttribs(self.info_)
        inpshape = self.info_[dgbkeys.inpshapedictstr]
        nrzin = inp.shape[-1]
        vertical =  isinstance(inpshape,int)
        swapaxes = False
        if vertical:
            nrzoutsamps = nrzin-inpshape+1
            nrpts = nrzoutsamps
        else:
            swapaxes = ((self.isflat_inlinemodel_ and self.applydir_ == dgbkeys.crosslinestr) or \
                        (self.isflat_xlinemodel_ and self.applydir_ == dgbkeys.inlinestr)) and not self.datais2d_
            if self.isflat_xlinemodel_:
                inpshape = (inpshape[1], inpshape[0], inpshape[2])
                self.info_[dgbkeys.inpshapedictstr] = inpshape
            if self.img2img_:
              if self.datais2d_:
                nrpts = inp.shape[0] if len(inp.shape) == 4 else 1
              else:
                nrpts = inp.shape[0] if len(inp.shape) == 5 else 1
            else:
              nrzoutsamps = nrzin - inpshape[2] +1
              nrpts = nrzoutsamps

        samples_shape = dgbhdf5.get_np_shape( inpshape, nrpts=nrpts,
                                              nrattribs=nrattribs )
        nrtrcs = samples_shape[-2]
        nrz = samples_shape[-1]
        if self.img2img_:
          if not self.datais2d_ and (self.isflat_inlinemodel_ or self.isflat_xlinemodel_):
            if nrpts == 1:
              samples = np.reshape( inp, (nrpts,*inp.shape) ).copy()
            else:
              samples = np.reshape( inp, inp.shape ).copy()
          else:
            samples = np.reshape( inp, samples_shape ).copy()
        else:
          allsamples = list()
          if nrz == 1:
              inp = np.transpose( inp )
              allsamples.append( np.resize( np.array(inp), samples_shape ) )
          else:
              loc_samples = np.empty( samples_shape, dtype=inp.dtype )
              if vertical:
                  for zidz in range(nrzoutsamps):
                      loc_samples[zidz,:,0,0,:] = inp[:,zidz:zidz+nrz]
                  allsamples.append( loc_samples )
              elif self.datais2d_:
                  for ich in range(inp.shape[0]):
                      for zidz in range(nrzoutsamps):
                          loc_samples[zidz] = inp[ich,:,zidz:zidz+nrz]
                  allsamples.append( loc_samples )
              else:
                  for zidz in range(nrzoutsamps):
                      loc_samples[zidz] = inp[:,:,:,zidz:zidz+nrz]
                  allsamples.append( loc_samples )

          samples = np.concatenate(allsamples)

        samples = self.preprocess( samples, swapaxes )

        ret = {}
        if self.img2img_ and not self.datais2d_ and \
            (self.isflat_inlinemodel_ or self.isflat_xlinemodel_) and \
            self.applydir_ in [dgbkeys.averagestr, dgbkeys.minstr, dgbkeys.maxstr]:
            ret[dgbkeys.preddictstr] = self.flatModelApply(inp, samples, samples_shape)
            if self.applydir_ in [dgbkeys.averagestr, dgbkeys.minstr, dgbkeys.maxstr]:
                samples = samples.swapaxes(*self._get_swapaxes_dim(samples))
                newret = self.flatModelApply(inp, samples, samples_shape)
                newret = newret.swapaxes(*self._get_swapaxes_dim(samples))
                if self.applydir_ == dgbkeys.averagestr:
                    ret[dgbkeys.preddictstr] = (newret + ret[dgbkeys.preddictstr])/2
                elif self.applydir_ == dgbkeys.minstr:
                    ret[dgbkeys.preddictstr] = np.minimum(newret, ret[dgbkeys.preddictstr])
                else :
                    ret[dgbkeys.preddictstr] = np.maximum(newret, ret[dgbkeys.preddictstr])
        else:
            ret = dgbmlapply.doApply( self.model_, self.info_, samples, \
                                      scaler=None, applyinfo=self.applyinfo_, \
                                      batchsize=self.batchsize_ )

        if dgbkeys.preddictstr in ret:
            ret[dgbkeys.preddictstr] = self.postprocess( ret[dgbkeys.preddictstr], swapaxes )
    
        res = list()
        outkeys = list()
        outkeys.append( dgbkeys.preddictstr )
        outkeys.append( dgbkeys.probadictstr )
        outkeys.append( dgbkeys.confdictstr )
        outkeys.append( dgbkeys.matchdictstr )
        for outkey in outkeys:
          if outkey in ret:
            res.append( ret[outkey] )

        return res

    def debug_msg(self,a,b=None,c=None,d=None,e=None,f=None,g=None,h=None):
        ret = str(a)
        if b != None:
          ret += ' '+str(b)
        if c != None:
          ret += ' '+str(c)
        if d != None:
          ret += ' '+str(d)
        if e != None:
          ret += ' '+str(e)
        if f != None:
          ret += ' '+str(f)
        if g != None:
          ret += ' '+str(g)
        if h != None:
          ret += ' '+str(h)
        if len(self.debugstr_) > 0:
          self.debugstr_ += '\n'
        self.debugstr_ += ret
        return self.debugstr_


class Message:
    def __init__(self, selector, sock, addr, applier):
        self._selector = selector
        self._sock = sock
        self._addr = addr
        self._recv_buffer = b""
        self._send_buffer = b""
        self._payload_len = None
        self._reqid = None
        self._subid = None
        self._jsonheader_len = None
        self._jsonheader = None
        self._request = None
        self._response_created = False
        self._applier = applier
        self._serverexception = None
        self._killreq = False

    def _set_selector_events_mask(self, mode):
        """Set selector to listen for events: mode is 'r', 'w', or 'rw'."""
        if mode == "r":
            events = selectors.EVENT_READ
        elif mode == "w":
            events = selectors.EVENT_WRITE
        elif mode == "rw":
            events = selectors.EVENT_READ | selectors.EVENT_WRITE
        else:
            raise ValueError(f"Invalid events mask mode {repr(mode)}.")
        self._selector.modify(self._sock, events, data=self)

    def _read(self):
        try:
            data = self._sock.recv(16777216)
        except BlockingIOError:
            return True # Resource temporarily unavailable (errno EWOULDBLOCK)

        if not data:
            self.close()
            return False

        self._recv_buffer += data
        return True

    def _write(self):
        if self._send_buffer:
            try:
                sent = self._sock.send(self._send_buffer)
            except BlockingIOError:
                return # Resource temporarily unavailable (errno EWOULDBLOCK)

            self._send_buffer = self._send_buffer[sent:]
            # Close when the buffer is drained. The response has been sent.
            if sent and not self._send_buffer:
                self.close()

    def _json_encode(self, obj, encoding):
        json_hdr = json.dumps(obj, ensure_ascii=False).encode(encoding)
        return struct.pack('=i',len(json_hdr)) + json_hdr

    def _json_decode(self, json_bytes, encoding):
        hdrlen = 4
        json_hdr = struct.unpack('=i',json_bytes[:hdrlen])[0]
        tiow = io.TextIOWrapper(
            io.BytesIO(json_bytes[hdrlen:hdrlen+json_hdr]), encoding=encoding, newline=""
        )
        obj = json.load(tiow)
        tiow.close()
        return (json_hdr,obj,json_bytes[hdrlen+json_hdr:])

    def _array_decode(self, arrptr, shapes, dtypes):
        offset = 0
        arrs = list()
        for shape,dtype in zip(shapes,dtypes):
          nrsamples = np.prod(shape,dtype=np.int64)
          arr = np.frombuffer(arrptr,dtype=dtype,count=nrsamples,offset=offset)
          arr = arr.reshape( shape )
          offset += arr.nbytes
          arrs.append( arr )
        return {
          'action': 'apply',
          'data': arrs
        }

    def _create_message(
        self, *, content_bytes, content_type, content_encoding, arrsize
    ):
        jsonheader = {
            "byteorder": sys.byteorder,
            "content-type": content_type,
            "content-encoding": content_encoding,
            "content-length": len(content_bytes),
        }
        if arrsize != None:
          jsonheader.update({ 'array-shape': arrsize })
        (self,jsonheader) = self._add_debug_str( jsonheader )
        jsonheader_bytes = self._json_encode(jsonheader, 'utf-8')
        payload = jsonheader_bytes + content_bytes
        od_hdr =   struct.pack('=i',len(payload)) \
                 + struct.pack('=i',self._reqid) \
                 + struct.pack('=h',self._subid)
        return od_hdr + payload

    def _make_exception_report(self, msg, exc):
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        stackstr = ''.join(tb.extract_tb(exc_tb,limit=10).format())
        return f'{msg}:\n{repr(exc)} on line {str(exc_tb.tb_lineno)} of script {fname}\n{stackstr}\n\n{self._applier.debugstr_}'

    def _create_response_json_content(self):
        action = self._request.get('action')
        content = { 'result': None }
        if action == 'status':
            content['result'] = 'Server online'
            content['pid'] = psutil.Process().pid
        elif action == 'kill':
            content['result'] = 'Kill request received'
            self._killreq = True
        elif action == 'parameters':
            try:
              self._applier.setParameters( self._request.get('value') )
            except Exception as e:
              self._serverexception = self._make_exception_report('Start error exception', e)
              content['result'] = self._serverexception
            else:
              content['result'] = 'Apply parameters received'
        else:
            content['result'] = f'Error: invalid action "{action}".'
        content_encoding = 'utf-8'
        response = {
            'content_bytes': self._json_encode(content, content_encoding),
            'content_type': 'text/json',
            'content_encoding': content_encoding,
            'arrsize': None,
        }
        return response

    def _create_response_array_content(self):
        action = self._request.get('action')
        try:
            res = list()
            if action == 'apply':
                for arr in self._request.get('data'):
                    res = self._applier.doWork(arr)
            else:
                content = {"result": f'Error: invalid action "{action}".'}
        except Exception as e:
            content = {'result': self._make_exception_report('Apply error exception', e)}
            self._applier.debugstr_ = ''
            content_encoding = 'utf-8'
            response = {
                'content_bytes': self._json_encode(content, content_encoding),
                'content_type': 'text/json',
                'content_encoding': content_encoding,
                'arrsize': None,
            }
            return (self,response)

        ret = bytes()
        dtypes = list()
        shapes = list()
        for arr in res:
          ret += arr.tobytes()
          shapes.append( arr.shape )
          dtypes.append( arr.dtype.name )
        response = {
          'content_bytes': ret,
          'content_type': "binary/array",
          'content_encoding': dtypes,
          'arrsize': shapes,
        }
        return (self,response)

    def _create_response_binary_content(self):
        response = {
            "content_bytes": b"First 10 bytes of request: "
            + self._request[:10],
            "content_type": "binary/custom-server-binary-type",
            "content_encoding": "binary",
            'arrsize': None,
        }
        return response

    def _add_debug_str( self, response ):
        if self._applier is None:
            return (self,response)
        debugstr_ = self._applier.debugstr_
        if len(debugstr_) > 0:
            response.update( {'debug-message': debugstr_} )
            self._applier.debugstr_ = ''
        return (self,response)

    def process_events(self, mask):
        if mask & selectors.EVENT_READ:
            self.read()
        if mask & selectors.EVENT_WRITE:
            self.write()

    def read(self):
        if not self._read():
            return

        if self._payload_len is None:
            self.process_odheader()

        if self._payload_len is not None:
            if self._jsonheader is None:
                self.process_jsonheader()

        if self._jsonheader:
            if self._request is None:
                self.process_request()

    def write(self):
        if self._request is not None:
            if not self._response_created:
                self.create_response()

        self._write()

    def close(self):
        try:
            self._selector.unregister(self._sock)
        except Exception:
            pass

        try:
            self._sock.close()
        except Exception:
            pass

        finally:
            # Delete reference to socket object for garbage collection
            self._sock = None

    def process_odheader(self):
        hdrlen = 10
        if len(self._recv_buffer) >= hdrlen:
            self._payload_len = struct.unpack('=i',self._recv_buffer[0:4])[0]
            self._reqid = struct.unpack('=i',self._recv_buffer[4:8])[0]
            self._subid = struct.unpack('=h',self._recv_buffer[8:hdrlen])[0]
            self._recv_buffer = self._recv_buffer[hdrlen:]

    def process_jsonheader(self):
        if len(self._recv_buffer) >= 4:
            (self._jsonheader_len,self._jsonheader,self._recv_buffer) = \
                self._json_decode(
                    self._recv_buffer, "utf-8"
            )
            for reqhdr in (
                "byteorder",
                "content-length",
                "content-type",
                "content-encoding",
            ):
                if reqhdr not in self._jsonheader:
                    raise ValueError(f'Missing required header "{reqhdr}".')

    def process_request(self):
        content_len = self._jsonheader["content-length"]
        if not len(self._recv_buffer) >= content_len:
            return
        data = self._recv_buffer[:content_len]
        self._recv_buffer = self._recv_buffer[content_len:]
        if self._jsonheader["content-type"] == "text/json":
            encoding = self._jsonheader["content-encoding"]
            (jsonsz,self._request,self._recv_buffer) = \
                                 self._json_decode(data, encoding)
        elif self._jsonheader["content-type"] == 'binary/array':
            shapes = self._jsonheader['array-shape']
            dtypes = self._jsonheader['content-encoding']
            self._request = self._array_decode(data,shapes,dtypes)
        else:
            # Binary or unknown content-type
            self._request = data
            print(
                f'received {self._jsonheader["content-type"]} request from',
                self._addr,
            )
        # Set selector to listen for write events, we're done reading.
        self._set_selector_events_mask('w')

    def create_response(self):
        if self._jsonheader["content-type"] == 'text/json':
            response = self._create_response_json_content()
        elif self._jsonheader["content-type"] == 'binary/array':
            (self,response) = self._create_response_array_content()
        else:
            # Binary or unknown content-type
            response = self._create_response_binary_content()
        message = self._create_message(**response)
        self._response_created = True
        self._send_buffer += message
