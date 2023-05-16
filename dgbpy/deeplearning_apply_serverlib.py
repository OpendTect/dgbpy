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
from dgbpy import dgbtorch
from dgbpy import dgbscikit, dgbkeras

class ExitCommand(Exception):
    pass

class ModelApplier:
    def __init__(self, modelfnm, applydir=dgbkeys.inlinestr, isfake=False):
        self.pars_ = None
        self.fakeapply_ = isfake
        self.scaler_ = None
        self.info_ = self._get_info(modelfnm)
        self.needtranspose_ = False
        self._set_transpose()
        self.model_ = None
        self.applyinfo_ = None
        self.batchsize_ = None
        self.debugstr = ''
        self.applydir_ = applydir

    def _get_info(self,modelfnm):
        info = dgbmlio.getInfo( modelfnm, quick=True )
        if self.fakeapply_:
            info[dgbkeys.plfdictstr] = dgbkeys.numpyvalstr
        return info

    def _set_transpose(self):
        if dgbhdf5.isSeisClass( self.info_ ) or \
            dgbhdf5.isImg2Img( self.info_ ):
            self.needtranspose_ = dgbhdf5.applyArrTranspose( self.info_ )
        else:
            self.needtranspose_ = False
    
    def setOutputs(self, outputs):
        if self.fakeapply_:
            self.applyinfo_ = dgbmlio.getApplyInfo( self.info_ )
        else:
            self.applyinfo_ = dgbmlio.getApplyInfo( self.info_, outputs )
        self.scaler_ = self.getScaler( outputs )
        if self.info_[dgbkeys.plfdictstr] == dgbkeys.kerasplfnm:
            if dgbkeys.prefercpustr in outputs:
                dgbkeras.set_compute_device( outputs[dgbkeys.prefercpustr] )
            if dgbkeras.defbatchstr in outputs:
                self.batchsize_ = outputs[dgbkeras.defbatchstr]
        elif self.info_[dgbkeys.plfdictstr] == dgbkeys.torchplfnm:
            if dgbkeys.prefercpustr in outputs:
                dgbtorch.set_compute_device( outputs[dgbkeys.prefercpustr] )
        if self.fakeapply_:
            return None
        modelfnm = self.info_[dgbkeys.filedictstr]
        (self.model_,self.info_) = dgbmlio.getModel( modelfnm, fortrain=False )
        self._set_transpose()

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
                    self.scaler_ = dgbscikit.getNewScaler( means, stddevs )

        return self.scaler_

    def preprocess(self,samples):
        if dgbhdf5.applyLocalStd( self.info_ ):
            self.scaler_ = dgbscikit.getScaler( samples, True )
        elif dgbhdf5.applyNormalization( self.info_ ):
            self.scaler_ = dgbscikit.getNewMinMaxScaler( samples )
        elif dgbhdf5.applyMinMaxScaling( self.info_ ):
            self.scaler_ = dgbscikit.getNewMinMaxScaler( samples, maxout=255 )
        elif dgbhdf5.applyGlobalStd( self.info_ ):
            if self.scaler_ == None:
                self.debugmsg_ = 'Missing scaler for global standardization' 
                raise TypeError

        if self.scaler_ != None:
            samples = dgbscikit.scale( samples, self.scaler_ )
        if self.needtranspose_:
            samples = np.transpose( samples, axes=(0,1,4,3,2) )
        return samples

    def postprocess(self,samples):
        if self.needtranspose_:
            samples = np.transpose( samples, axes=(0,1,4,3,2) )

        if self.is2dinp_ and len(samples.shape)==5:
            samples = samples[:,:,samples.shape[2]//2,:,:]
        elif self.swapaxes_:
            samples = samples.swapaxes(2, 3)
        
        if dgbhdf5.unscaleOutput( self.info_ ):
            if self.scaler_:
                samples = dgbscikit.unscale( samples, self.scaler_ )

        return samples

    def flatModelApply(self, inp, samples, samples_shape):
        inpshape = self.info_[dgbkeys.inpshapedictstr]
        outshape = (1,) + inp.shape
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

    def doWork(self,inp):
        nrattribs = inp.shape[0]
        inpshape = self.info_[dgbkeys.inpshapedictstr]
        nrzin = inp.shape[-1]
        vertical =  isinstance(inpshape,int)
        self.isflat_inlinemodel_ = False
        self.isflat_xlinemodel_ = False
        self.is2dinp_ = False
        self.is3dmodel_ = False
        self.swapaxes_ = False
        if vertical:
            nrzoutsamps = nrzin-inpshape+1
        else:
            self.is2dinp_ = len(inp.shape) == 3
            self.is3dmodel_ = len(inpshape) == 3
            self.isflat_inlinemodel_ = self.is3dmodel_ and inpshape[0] == 1 and inpshape[1] > 1
            self.isflat_xlinemodel_ = self.is3dmodel_ and inpshape[0] > 1 and inpshape[1] == 1
            self.swapaxes_ = (self.isflat_inlinemodel_ or self.isflat_xlinemodel_) and self.applydir_ == dgbkeys.crosslinestr \
                             and not self.is2dinp_
            if self.isflat_xlinemodel_:
                inpshape = (inpshape[1], inpshape[0], inpshape[2])
                self.info_[dgbkeys.inpshapedictstr] = inpshape

            nrzoutsamps = nrzin - inpshape[2] +1
        samples_shape = dgbhdf5.get_np_shape( inpshape, nrattribs=nrattribs,
                                              nrpts=nrzoutsamps )
        nrtrcs = samples_shape[-2]
        nrz = samples_shape[-1]
        allsamples = list()
        if nrz == 1:
            inp = np.transpose( inp )
            allsamples.append( np.resize( np.array(inp), samples_shape ) )
        else:
            loc_samples = np.empty( samples_shape, dtype=inp.dtype )
            if (self.isflat_inlinemodel_ or self.isflat_xlinemodel_) and not self.is2dinp_:
                loc_samples = np.empty((1,)+inp.shape, dtype=inp.dtype)

            if vertical:
                for zidz in range(nrzoutsamps):
                    loc_samples[zidz,:,0,0,:] = inp[:,zidz:zidz+nrz]
                allsamples.append( loc_samples )
            elif self.is2dinp_:
                for zidz in range(nrzoutsamps):
                  loc_samples[zidz] = inp[:,:,zidz:zidz+nrz]
                allsamples.append( loc_samples )
            else :
                for zidz in range(nrzoutsamps):
                    loc_samples[zidz] = inp[:,:,:,zidz:zidz+nrz]
                allsamples.append( loc_samples )

        samples = np.concatenate(allsamples)
        if self.swapaxes_:
            samples = samples.swapaxes(2, 3)

        samples = self.preprocess( samples )

        ret = {}
        if self.info_[dgbkeys.learntypedictstr] == dgbkeys.seisimgtoimgtypestr and not self.is2dinp_ and \
            (self.isflat_inlinemodel_ or self.isflat_xlinemodel_):
            ret[dgbkeys.preddictstr] = self.flatModelApply(inp, samples, samples_shape)
            if self.applydir_ in [dgbkeys.averagestr, dgbkeys.minstr, dgbkeys.maxstr]:
                samples = samples.swapaxes(2, 3)
                newret = self.flatModelApply(inp, samples, samples_shape)
                newret = newret.swapaxes(2, 3)
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
            ret[dgbkeys.preddictstr] = self.postprocess( ret[dgbkeys.preddictstr] )
    
        res = list()
        outkeys = list()
        outkeys.append( dgbkeys.preddictstr )
        outkeys.append( dgbkeys.probadictstr )
        outkeys.append( dgbkeys.confdictstr )
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
        if len(self.debugstr) > 0:
          self.debugstr += '\n'
        self.debugstr += ret
        return self.debugstr


class Message:
    def __init__(self, selector, sock, addr, applier):
        self.selector = selector
        self.sock = sock
        self.addr = addr
        self._recv_buffer = b""
        self._send_buffer = b""
        self._payload_len = None
        self._reqid = None
        self._subid = None
        self._jsonheader_len = None
        self.jsonheader = None
        self.request = None
        self.response_created = False
        self.applier = applier
        self.lastmessage = False

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
        self.selector.modify(self.sock, events, data=self)

    def _read(self):
        try:
            # Should be ready to read
            data = self.sock.recv(16777216)
        except BlockingIOError:
            # Resource temporarily unavailable (errno EWOULDBLOCK)
            pass
        else:
            if data:
                self._recv_buffer += data
            else:
                raise RuntimeError("Peer closed.")

    def _write(self):
        if self._send_buffer:
            try:
                # Should be ready to write
                sent = self.sock.send(self._send_buffer)
            except BlockingIOError:
                # Resource temporarily unavailable (errno EWOULDBLOCK)
                pass
            else:
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

    def _create_response_json_content(self):
        action = self.request.get('action')
        content = { 'result': None }
        if action == 'status':
            content['result'] = 'Server online'
            content['pid'] = psutil.Process().pid
        elif action == 'kill':
            content['result'] = 'Kill request received'
            self.lastmessage = True
        elif action == 'outputs':
            try:
              self.applier.setOutputs( self.request.get('value') )
              content['result'] = 'Output names received'
            except Exception as e:
              content = {"result": f'start error exception: {repr(e)}.'}
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
        action = self.request.get('action')
        try:
            res = list()
            if action == 'apply':
                for arr in self.request.get('data'):
                    res = self.applier.doWork(arr)
            else:
                content = {"result": f'Error: invalid action "{action}".'}
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            stackstr = ''.join(tb.extract_tb(exc_tb,limit=10).format())
            content = {'result': f'Apply error exception:\n{repr(e)+" on line "+str(exc_tb.tb_lineno)+" of script "+fname}\n{stackstr}\n\n{self.applier.debugstr}'}
            self.applier.debugstr = ''
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
            + self.request[:10],
            "content_type": "binary/custom-server-binary-type",
            "content_encoding": "binary",
            'arrsize': None,
        }
        return response

    def _add_debug_str( self, response ):
        if self.applier == None:
            return (self,response)
        debugstr = self.applier.debugstr
        if len(debugstr) > 0:
            response.update( {'debug-message': debugstr} )
            self.applier.debugstr = ''
        return (self,response)

    def process_events(self, mask):
        if mask & selectors.EVENT_READ:
            self.read()
        if mask & selectors.EVENT_WRITE:
            self.write()

    def read(self):
        self._read()

        if self._payload_len is None:
            self.process_odheader()

        if self._payload_len is not None:
            if self.jsonheader is None:
                self.process_jsonheader()

        if self.jsonheader:
            if self.request is None:
                self.process_request()

    def write(self):
        if self.request:
            if not self.response_created:
                self.create_response()

        self._write()

    def close(self):
        try:
            self.selector.unregister(self.sock)
        except Exception as e:
            print(
                f"error: selector.unregister() exception for",
                f"{self.addr}: {repr(e)}",
            )

        try:
            self.sock.close()
        except OSError as e:
            print(
                f"error: socket.close() exception for",
                f"{self.addr}: {repr(e)}",
            )
        finally:
            # Delete reference to socket object for garbage collection
            self.sock = None

    def process_odheader(self):
        hdrlen = 10
        if len(self._recv_buffer) >= hdrlen:
            self._payload_len = struct.unpack('=i',self._recv_buffer[0:4])[0]
            self._reqid = struct.unpack('=i',self._recv_buffer[4:8])[0]
            self._subid = struct.unpack('=h',self._recv_buffer[8:hdrlen])[0]
            self._recv_buffer = self._recv_buffer[hdrlen:]

    def process_jsonheader(self):
        if len(self._recv_buffer) >= 4:
            (self._jsonheader_len,self.jsonheader,self._recv_buffer) = \
                self._json_decode(
                    self._recv_buffer, "utf-8"
            )
            for reqhdr in (
                "byteorder",
                "content-length",
                "content-type",
                "content-encoding",
            ):
                if reqhdr not in self.jsonheader:
                    raise ValueError(f'Missing required header "{reqhdr}".')

    def process_request(self):
        content_len = self.jsonheader["content-length"]
        if not len(self._recv_buffer) >= content_len:
            return
        data = self._recv_buffer[:content_len]
        self._recv_buffer = self._recv_buffer[content_len:]
        if self.jsonheader["content-type"] == "text/json":
            encoding = self.jsonheader["content-encoding"]
            (jsonsz,self.request,self._recv_buffer) = \
                                 self._json_decode(data, encoding)
        elif self.jsonheader["content-type"] == 'binary/array':
            shapes = self.jsonheader['array-shape']
            dtypes = self.jsonheader['content-encoding']
            self.request = self._array_decode(data,shapes,dtypes)
        else:
            # Binary or unknown content-type
            self.request = data
            print(
                f'received {self.jsonheader["content-type"]} request from',
                self.addr,
            )
        # Set selector to listen for write events, we're done reading.
        self._set_selector_events_mask("w")

    def create_response(self):
        if self.jsonheader["content-type"] == 'text/json':
            response = self._create_response_json_content()
        elif self.jsonheader["content-type"] == 'binary/array':
            (self,response) = self._create_response_array_content()
        else:
            # Binary or unknown content-type
            response = self._create_response_binary_content()
        message = self._create_message(**response)
        self.response_created = True
        self._send_buffer += message
