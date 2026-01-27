#!/usr/bin/env python3
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
import numpy as np
import os
import psutil
import selectors
import signal
import socket
import threading
import traceback

from odpy.common import initLogging, redirect_stdout, std_msg, log_msg, Timer
import dgbpy.keystr as dgbkeys
import dgbpy.deeplearning_apply_serverlib as applylib


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
            description='Server application of a trained machine learning model')
  parser.add_argument( '-v', '--version',
                       action='version',version='%(prog)s 3.0' )
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
  loggrp = parser.add_argument_group( 'Logging' )
  loggrp.add_argument( '--log',
              dest='logfile', metavar='file', nargs='?',
              type=argparse.FileType('w'), default=sys.stdout,
              help='Progress report output' )
  loggrp.add_argument( '--syslog',
              dest='sysout', metavar='stdout', nargs='?',
              type=argparse.FileType('w'), default=sys.stdout,
              help='System log' )
  return parser.parse_args()

def start_server( args ):
  # Start listening as quickly as possible
  sel = selectors.DefaultSelector()
  host, port = args['addr'], args['port']
  if host == 'LOCAL':
    addr = str(port)
    try:
      os.unlink(addr)
    except OSError:
      if os.path.exists(addr):
        raise
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
    log_msg( f'Connection error for port {port} on host {host}' )
    log_msg( e )
    raise e
  std_msg("listening on", addr)
  lsock.setblocking(False)
  sel.register(lsock, selectors.EVENT_READ, data=None)
  return sel

def signal_handler(signal, frame):
  raise applylib.ExitCommand()

def timerCB( parentproc ):
  if not parentproc.is_running():
    raise applylib.ParentExitCommand

def accept_wrapper(sel, applylib, sock, applier):
  conn, addr = sock.accept()  # Should be ready to read
  conn.setblocking(False)
  message = applylib.Message(sel, conn, addr, applier)
  sel.register(conn, selectors.EVENT_READ, data=message)

def main() -> int:
  args = vars(parse_args())

  initLogging( args )
  redirect_stdout()

  sel = start_server( args )

  # Keep all lengthy operations below
  signal.signal(signal.SIGTERM,signal_handler)

  timer = None
  parentproc = None
  if 'parentpid' in args:
    ppid = args['parentpid']
    if ppid > 0:
      parentproc = psutil.Process( ppid )
      timer = Timer(15, timerCB, args=(parentproc,))
      timer.start()

  applier = None
  ex_code = 0
  try:
    if applier == None:
      applier = applylib.ModelApplier( args['modelfile'].name )
    killreq = False
    servererror = None
    parentisdead = False
    while not killreq and servererror is None and not parentisdead:
      events = sel.select(timeout=300)
      for key, mask in events:
        if key.data is None:
          accept_wrapper(sel, applylib, key.fileobj, applier)
        else:
          message = key.data
          try:
            message.process_events(mask)
          except Exception:
            log_msg( "main: error: exception for",
                    f"{message._addr}:\n{traceback.format_exc()}",
            )
            ex_code = 1
            message.close()
          finally:
            if parentproc != None and not parentproc.is_running():
              parentisdead = True
              log_msg('Found dead parent, exiting')
              if timer is not None:
                timer.cancel()

            servererror = message._serverexception
            if mask & selectors.EVENT_WRITE:
              killreq = message._killreq

            if servererror is not None or killreq or parentisdead or ex_code != 0:
              break
            applier = message._applier
  except KeyboardInterrupt:
    log_msg('caught keyboard interrupt, exiting')
    ex_code = 130
  except applylib.ExitCommand:
    log_msg('Shutdown requested, exiting')
  except applylib.ParentExitCommand:
    log_msg('Found dead parent, exiting')
    ex_code = 2
  finally:
    if timer is not None:
      timer.cancel()
    sel.close()
  return ex_code

if __name__ == '__main__':
  sys.exit(main())

