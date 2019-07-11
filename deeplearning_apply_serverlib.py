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
import psutil
import selectors
import struct
import sys

from odpy.common import *
import dgbpy.keystr as dgbkeys
from dgbpy import hdf5 as dgbhdf5
from dgbpy import mlio as dgbmlio
from dgbpy import mlapply as dgbmlapply

class ExitCommand(Exception):
    pass

class ModelApplier:
    def __init__(self, modelfnm,isfake=False):
        self.fnm_ = modelfnm
        self.pars_ = None
        self.fakeapply_ = isfake
        (self.model_,self.info_,self.scaler_) = self._open()
        self.applyinfo_ = None

    def _open(self):
        modelfnm = self.fnm_
        if self.fakeapply_:
            info = dgbmlio.getInfo( modelfnm )
            info[dgbkeys.plfdictstr] = dgbkeys.numpyvalstr
            return (None,info,None)
        else:
            return dgbmlio.getModel( modelfnm )

    def setOutputs(self, outputs):
        if self.fakeapply_:
            self.applyinfo_ = dgbmlio.getApplyInfo( self.info_ )
        else:
            self.applyinfo_ = dgbmlio.getApplyInfo( self.info_, outputs )

    def _usePar(self, pars):
        self.pars_ = pars

    def hasModel(self):
        return self.model_ != None

    def doWork(self,inp):
        nrattribs = inp.shape[0]
        stepout = self.info_['stepout']
        nrzin = inp.shape[-1]
        vertical =  isinstance(stepout,int)
        if vertical:
          nroutsamps = nrzin - 2*stepout
        else:
          nroutsamps = nrzin - 2*stepout[2]
        samples_shape = dgbhdf5.get_np_shape( stepout, nrattribs=nrattribs,
                                              nrpts=nroutsamps )
        nrz = samples_shape[-1]
        if nrz == 1:
          samples = np.resize( np.array(inp), samples_shape )
        else:
          samples = np.empty( samples_shape, dtype=inp.dtype )
          if vertical:
            for zidz in range(nroutsamps):
              samples[zidz,:,0,0,:] = inp[:,zidz:zidz+nrz]
          else:
            for zidz in range(nroutsamps):
              samples[zidz] = inp[:,:,:,zidz:zidz+nrz]
        ret = dgbmlapply.doApply( self.model_, self.info_, samples, \
                                  applyinfo=self.applyinfo_ )
        res = list()
        if dgbkeys.preddictstr in ret:
          res.append( ret[dgbkeys.preddictstr] )
        if dgbkeys.probadictstr in ret:
          res.append( ret[dgbkeys.probadictstr] )
        if dgbkeys.confdictstr in ret:
          res.append( ret[dgbkeys.confdictstr] )
        return res


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
          nrsamples = np.prod(shape)
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
            self.applier.setOutputs( self.request.get('value') )
            content['result'] = 'Output names received'
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
        res = list()
        if action == 'apply':
            try:
                for arr in self.request.get('data'):
                    res = self.applier.doWork(arr)
            except Exception as e:
                content = {"result": f'Apply error exception: {repr(e)}.'}
                content_encoding = 'utf-8'
                response = {
                    'content_bytes': self._json_encode(content, content_encoding),
                    'content_type': 'text/json',
                    'content_encoding': content_encoding,
                    'arrsize': None,
                }
                return response
        else:
            content = {"result": f'Error: invalid action "{action}".'}
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
        return response

    def _create_response_binary_content(self):
        response = {
            "content_bytes": b"First 10 bytes of request: "
            + self.request[:10],
            "content_type": "binary/custom-server-binary-type",
            "content_encoding": "binary",
            'arrsize': None,
        }
        return response

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
            response = self._create_response_array_content()
        else:
            # Binary or unknown content-type
            response = self._create_response_binary_content()
        message = self._create_message(**response)
        self.response_created = True
        self._send_buffer += message
