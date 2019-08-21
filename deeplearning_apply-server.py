#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : A. Huck
# DATE     : June 2019
#
# Deep learning apply server
#
#

import argparse
import numpy as np
import os
import psutil
import selectors
import signal
import socket
import sys
import time
import threading
import traceback

from odpy.common import *
from odpy import oscommand
import dgbpy.deeplearning_apply_serverlib as applylib

sel = selectors.DefaultSelector()

# -- command line parser

parser = argparse.ArgumentParser(
          description='Server application of a trained machine learning model')
parser.add_argument( '-v', '--version',
            action='version',version='%(prog)s 2.0')
parser.add_argument( 'modelfile', type=argparse.FileType('r'),
                     help='The input trained model file' )
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


args = vars(parser.parse_args())
initLogging( args )

redirect_stdout()

applier = None
pid = psutil.Process().pid

def signal_handler(signal, frame):
  raise applylib.ExitCommand()
signal.signal(signal.SIGINT,signal_handler)

def timerCB():
  if not parentproc.is_running():
    os.kill( pid, signal.SIGINT )

def accept_wrapper(sock,applier):
  conn, addr = sock.accept()  # Should be ready to read
  conn.setblocking(True)
  message = applylib.Message(sel, conn, addr, applier)
  sel.register(conn, selectors.EVENT_READ, data=message)

host,port = args['addr'], args['port']
lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Avoid bind() exception: OSError: [Errno 48] Address already in use
lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
lsock.bind((host, port))
lsock.listen()
std_msg("listening on", (host, port))
lsock.setblocking(True)
sel.register(lsock, selectors.EVENT_READ, data=None)

parentproc = psutil.Process().parent()
maxdepth = 10
while maxdepth > 0:
  parentproc = parentproc.parent()
  try:
    pname = parentproc.name() 
  except AttributeError:
    parentproc = psutil.Process().parent()
    break
  if 'od_main' in pname or 'od_deeplearn' in pname:
    break
  maxdepth -= 1

timer = threading.Timer(15, timerCB)
timer.start()

try:
  if applier == None:
    applier = applylib.ModelApplier( args['modelfile'].name, args['fakeapply'] )
  lastmessage = False
  cont = True
  while cont:
    events = sel.select(timeout=None)
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
        cont = cont and parentproc.is_running()
        if cont:
          applier = message.applier
except KeyboardInterrupt:
  std_msg('caught keyboard interrupt, exiting')
except applylib.ExitCommand:
  std_msg('Found dead parent, exiting')
finally:
  timer.cancel()
  sel.close()
