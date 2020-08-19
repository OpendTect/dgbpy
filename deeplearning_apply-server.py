#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : A. Huck
# DATE     : June 2019
#
# Deep learning apply server
#
#

import sys
import argparse

parser = argparse.ArgumentParser(
          description='Server application of a trained machine learning model')
parser.add_argument( '-v', '--version',
            action='version',version='%(prog)s 2.0')
parser.add_argument( 'modelfile', type=argparse.FileType('r'),
                     help='The input trained model file' )
parser.add_argument( '--ppid',
                     dest='parentpid', action='store',
                     type=int, default=-1,
                     help='PID of the parent process' )
netgrp = parser.add_argument_group( 'Network' )
netgrp.add_argument( '--address',
            dest='addr', metavar='ADDRESS', action='store',
            type=str, default='localhost',
            help='Address to listen on' )
netgrp.add_argument( '--port',
            dest='port', action='store',
            type=int, default=65432,
            help='Port to listen on')
datagrp = parser.add_argument_group( 'Data' )
datagrp.add_argument( '--mldir',
            dest='mldir', nargs=1,
            help='Machine Learning Directory' )
loggrp = parser.add_argument_group( 'Logging' )
loggrp.add_argument( '--log',
            dest='logfile', metavar='file', nargs='?',
            type=argparse.FileType('w'), default=sys.stdout,
            help='Progress report output' )
loggrp.add_argument( '--syslog',
            dest='sysout', metavar='stdout', nargs='?',
            type=argparse.FileType('w'), default=sys.stdout,
            help='System log' )
# optional
parser.add_argument( '--fakeapply', dest='fakeapply', action='store_true',
                     default=False,
                     help="applies a numpy average instead of the model" )
parser.add_argument( '--local', dest='localserv', action='store_true',
                     default=False,
                     help="use a local network socket connection" )

args = vars(parser.parse_args())
from odpy.common import *
initLogging( args )
redirect_stdout()

# Start listening as quickly as possible
import selectors
import socket
sel = selectors.DefaultSelector()
local = args['localserv']
host,port = args['addr'], args['port']
if local:
  addr = str(port)
  try:
    os.unlink(addr)
  except OSError:
    if os.path.exists(addr):
      raise
  host = 'LOCAL'
  sockfam = socket.AF_UNIX
else:
  addr = (host, port)
  sockfam = socket.AF_INET
lsock = socket.socket(sockfam, socket.SOCK_STREAM)
# Avoid bind() exception: OSError: [Errno 48] Address already in use
#lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
  lsock.bind(addr)
  lsock.listen()
except Exception as e:
  log_msg( 'Connection error for port', port, 'on host', host )
  log_msg( e )
  raise e
std_msg("listening on", addr)
lsock.setblocking(True)
sel.register(lsock, selectors.EVENT_READ, data=None)


# Keep all lengthy operations below
import numpy as np
import os
import psutil
import signal
import threading
import traceback

from odpy.common import Timer
import dgbpy.deeplearning_apply_serverlib as applylib


def signal_handler(signal, frame):
  raise applylib.ExitCommand()
signal.signal(signal.SIGINT,signal_handler)

def timerCB():
  if not parentproc.is_running():
    os.kill( psutil.Process().pid, signal.SIGINT )

def accept_wrapper(sock,applier):
  conn, addr = sock.accept()  # Should be ready to read
  conn.setblocking(True)
  message = applylib.Message(sel, conn, addr, applier)
  sel.register(conn, selectors.EVENT_READ, data=message)

timer = Timer(15, timerCB)
parentproc = None
if 'parentpid' in args:
  ppid = args['parentpid']
  if ppid > 0:
    parentproc = psutil.Process( ppid )
    timer.start()

applier = None
try:
  if applier == None:
    applier = applylib.ModelApplier( args['modelfile'].name, args['fakeapply'] )
  lastmessage = False
  cont = True
  while cont:
    events = sel.select(timeout=300)
    for key, mask in events:
      if key.data is None:
        accept_wrapper(key.fileobj,applier)
      else:
        message = key.data
        try:
          message.process_events(mask)
        except Exception:
          log_msg( "main: error: exception for",
                  f"{message.addr}:\n{traceback.format_exc()}",
          )
          message.close()
        lastmessage = lastmessage or message.lastmessage
        cont = not lastmessage or len(events) > 1
        cont = cont
        if parentproc != None and not parentproc.is_running():
          cont = False
        if cont:
          applier = message.applier
except KeyboardInterrupt:
  std_msg('caught keyboard interrupt, exiting')
except applylib.ExitCommand:
  std_msg('Found dead parent, exiting')
finally:
  timer.cancel()
  sel.close()
