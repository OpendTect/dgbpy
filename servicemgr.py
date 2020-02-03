#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : Wayne Mogg
# DATE     : January 2020
#
# Service Manager
#
# 
import json
import os
import psutil
import signal
import socket
import struct
import sys
import threading
import odpy.common as odcommon

class ServiceMgr:
  def __init__(self, cmdserver, ppid):
    self.host = None
    self.port = None
    if '@' in cmdserver:
      info = cmdserver.split('@')
      cmdserver = info[1]
    if ':' in cmdserver:
      info = cmdserver.split(':')
      self.host = info[0]
      self.port = int(info[1])

    self._parentproc = None
    if ppid > 0:
      self._parentproc = psutil.Process(ppid)
      self._parentChkTimer = odcommon.Timer(15, self._parentChkCB)
      self._parentChkTimer.start()

    self._actions = dict()
    
  def __enter__(self):
    return self
  
  def __exit__(self, exc_type, exc_value, traceback):
    if self._parentproc != None:
      self._parentChkTimer.cancel()
    
  def _parentChkCB(self):
    if self._parentproc != None and not self._parentproc.is_running():
      odcommon.std_msg('Found dead parent, exiting')
      self.stop()
      os.kill(psutil.Process().pid, signal.SIGINT)

  def sendEvent(self, eventstr):
    with Connection(self.host, self.port) as conn:
      packet = Packet()
      packet.setIsNewRequest()
      action = {'action': eventstr}
      packet.setTextPayload(action)
      conn.sendPacket(packet)

  def sendObject(self, objkey, jsonobj):
    with Connection(self.host, self.port) as conn:
      packet = Packet()
      packet.setIsNewRequest()
      obj = dict()
      obj[objkey] = jsonobj
      packet.setTextPayload(obj)
      conn.sendPacket(packet)
    
class Connection:
  def __init__(self, host, port):
    self.addr = (host, port)
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.sock.setblocking(False)
    self.sock.connect_ex(self.addr)
    
  def __enter__(self):
    return self
  
  def __exit__(self, exc_type, exc_value, traceback):
    self.sock.close()
  
  def sendPacket(self, packet):
    pack = packet.packet
    while len(pack):
      try:
        sent = self.sock.send(pack)
      except BlockingIOError:
        pass
      else:
        pack = pack[sent:]
      

class Packet:
  _curreqid = 0
  def __init__(self):
    self._reqid = None
    self._subid = None
    self._jsonheader_len = None
    self.jsonheader = None
    self.packet = None
    self._lock = threading.Lock()

  def setIsNewRequest(self):
    with self._lock:
      self._curreqid += 1
    self._reqid = self._curreqid
    self._subid = -1
    return self._reqid
    
  def setTextPayload(self, jsonobj):
    content_encoding = 'utf-8'
    payload = {
          'content_bytes': self._json_encode(jsonobj, content_encoding),
          'content_type': 'text/json',
          'content_encoding': content_encoding,
          'arrsize': None,
    }
    self._createPacket(payload)

  def _createPacket(self, payload):
    jsonheader = {
        'byteorder': sys.byteorder,
        'content-type': payload['content_type'],
        'content-encoding': payload['content_encoding'],
        'content-length': len(payload['content_bytes']),
        }
    if payload['arrsize'] != None:
      jsonheader.update({ 'array-shape': payload['arrsize'] })

    jsonheader_bytes = self._json_encode(jsonheader, 'utf-8')
    payload_bytes = jsonheader_bytes + payload['content_bytes']
    od_hdr = struct.pack('=i',len(payload_bytes)) \
             + struct.pack('=i',self._reqid) \
             + struct.pack('=h',self._subid)
    self.packet = od_hdr + payload_bytes

  def _json_encode(self, obj, encoding):
    json_hdr = json.dumps(obj, ensure_ascii=False).encode(encoding)
    return struct.pack('=i',len(json_hdr)) + json_hdr
