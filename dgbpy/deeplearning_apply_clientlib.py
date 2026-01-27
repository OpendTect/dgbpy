#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : A. Huck
# DATE     : June 2019
#
# Deep learning apply server
#
#

import sys
import selectors
import json
import io
import numpy as np
import struct

from odpy.common import *


class Message:
    def __init__(self, selector, sock, addr, request):
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
        self._response = None
        self._request = request
        self._request_queued = False
        self._serverpid = -1
        self._serverexception = None
        self._parshandled = False
        self._killreqhandled = False

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
                # Resource temporarily unavailable (errno EWOULDBLOCK)
                pass
            else:
                self._send_buffer = self._send_buffer[sent:]

    def _json_encode(self, obj, encoding):
        json_hdr = json.dumps(obj, ensure_ascii=False).encode(encoding)
        return struct.pack('=i',len(json_hdr)) + json_hdr

    def _json_decode(self, json_bytes, encoding):
        json_hdr = struct.unpack('=i',json_bytes[:4])[0]
        tiow = io.TextIOWrapper(
            io.BytesIO(json_bytes[4:4+json_hdr]), encoding=encoding, newline=""
        )
        obj = json.load(tiow)
        tiow.close()
        return (obj,json_bytes[4+json_hdr:])

    def _array_encode(self, objs):
        ret = bytes()
        shapes = list()
        for obj in objs:
          ret += obj.tobytes()
          shapes.append( obj.shape )
        return (ret,shapes)

    def _array_decode(self, arrptr, shapes, dtypes):
        ret = list()
        offset = 0
        for shape,dtype in zip(shapes,dtypes):
          nrsamples = np.prod(shape,dtype=np.int64)
          arr = np.frombuffer(arrptr,dtype,count=nrsamples,offset=offset)
          arr = arr.reshape( shape )
          offset += arr.nbytes
          ret.append( arr )
        return {
          'result': 'arrays',
          'data': ret
        }

    def _create_message(
        self, *, content_bytes, content_type, content_encoding, arrsize
    ):
        jsonheader = {
            'byteorder': sys.byteorder,
            'content-type': content_type,
            'content-encoding': content_encoding,
            'content-length': len(content_bytes),
        }
        if arrsize != None:
          jsonheader.update({ 'array-shape': arrsize })
        jsonheader_bytes = self._json_encode(jsonheader, 'utf-8')
        od_hdr =   struct.pack('=i',len(jsonheader_bytes)+len(content_bytes)) \
                 + struct.pack('=i',1) \
                 + struct.pack('=h',-1)
        message = od_hdr + jsonheader_bytes + content_bytes
        return message

    def _process_response_json_content(self):
        content = self._response
        result = content.get('result')
        if result == 'Server online':
            self._serverpid = content.get('pid')
        elif result.startswith('Start error exception'):
            self._serverexception = result
        elif result.startswith('Apply error exception'):
            self._serverexception = result
        elif result == 'Apply parameters received':
            self._parshandled = True
        elif result == 'Kill request received':
            self._killreqhandled = True

    def _process_response_array_content(self):
        content = self._response
        result = content.get('result')

    def _process_response_binary_content(self):
        content = self._response

    def process_events(self, mask):
        if mask & selectors.EVENT_READ:
            self.read()
        if mask & selectors.EVENT_WRITE:
            self.write()

    def read(self):
        if not self._read():
            return

        if self._jsonheader_len is None:
            self.process_protoheader()

        if self._jsonheader_len is not None:
            if self._jsonheader is None:
                self.process_jsonheader()

        if self._jsonheader:
            if self._response is None:
                self.process_response()

    def write(self):
        if not self._request_queued:
            self.queue_request()

        self._write()

        if self._request_queued:
            if not self._send_buffer:
                # Set selector to listen for read events, we're done writing.
                self._set_selector_events_mask('r')

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

    def queue_request(self):
        content = self._request['content']
        content_type = self._request['type']
        content_encoding = self._request['encoding']
        if content_type == 'text/json':
            req = {
                'content_bytes': self._json_encode(content, content_encoding),
                'content_type': content_type,
                'content_encoding': content_encoding,
                'arrsize': None,
            }
        elif content_type == 'binary/array':
            (arrsptr,shapes) = self._array_encode(content)
            req = {
              'content_bytes': arrsptr,
              'content_type': content_type,
              'content_encoding': content_encoding,
              'arrsize': shapes,
            }
        else:
            req = {
                'content_bytes': content,
                'content_type': content_type,
                'content_encoding': content_encoding,
                'arrsize': None,
            }
        message = self._create_message(**req)
        self._send_buffer += message
        self._request_queued = True

    def process_protoheader(self):
        hdrlen = 10
        if len(self._recv_buffer) >= hdrlen:
          self._payload_len = struct.unpack('=i',self._recv_buffer[0:4])[0]
          self._reqid = struct.unpack('=i',self._recv_buffer[4:8])[0]
          self._subid = struct.unpack('=h',self._recv_buffer[8:hdrlen])[0]
          self._jsonheader_len = struct.unpack('=i',self._recv_buffer[hdrlen:14])[0]
          self._recv_buffer = self._recv_buffer[hdrlen:]

    def process_jsonheader(self):
        hdrlen = self._jsonheader_len
        if len(self._recv_buffer) >= hdrlen:
            (self._jsonheader,self._recv_buffer) = self._json_decode(
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

    def process_response(self):
        content_len = self._jsonheader["content-length"]
        if not len(self._recv_buffer) >= content_len:
            return
        data = self._recv_buffer[:content_len]
        self._recv_buffer = self._recv_buffer[content_len:]
        if self._jsonheader["content-type"] == "text/json":
            encoding = self._jsonheader["content-encoding"]
            (self._response,self._recv_buffer) = self._json_decode(data, encoding)
            self._process_response_json_content()
        elif self._jsonheader["content-type"] == 'binary/array':
            shapes = self._jsonheader['array-shape']
            dtypes = self._jsonheader['content-encoding']
            self._response = self._array_decode(data,shapes,dtypes)
            self._process_response_array_content()
        else:
            # Binary or unknown content-type
            self._response = data
            log_msg(
                f'received {self._jsonheader["content-type"]} response from',
                self._addr,
            )
            self._process_response_binary_content()
        # Close when response has been processed
        self.close()
