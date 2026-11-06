#!/usr/bin/env python3
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : M. Fazelzadeh
# DATE     : Oct 2026
#
# Deep learning train server
#
#

import sys
import argparse
import os
import psutil
import selectors
import signal
import socket
import traceback

from odpy.common import initLogging, redirect_stdout, std_msg, log_msg, Timer
import dgbpy.deeplearning_train_serverlib as trainlib


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
            description='Server application for ML training')
  parser.add_argument( '-v', '--version',
                       action='version',version='%(prog)s 1.0' )
  parser.add_argument( 'modelfile', type=argparse.FileType('r'),
                       help='The input pretrained model file' )
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
              type=int, default=65482,
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
  raise trainlib.ExitCommand()

def timerCB( parentproc ):
  if not parentproc.is_running():
    raise trainlib.ParentExitCommand

def accept_wrapper(sel, sock, trainer):
  conn, addr = sock.accept()  # Should be ready to read
  conn.setblocking(False)
  message = trainlib.Message(sel, conn, addr, trainer)
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

  trainer = None
  ex_code = 0
  try:
    trainer = trainlib.ModelTrainer( args['modelfile'].name )
    killreq = False
    servererror = None
    parentisdead = False
    while not killreq and servererror is None and not parentisdead and ex_code == 0:
      events = sel.select(timeout=300)
      for key, mask in events:
        if key.data is None:
          accept_wrapper(sel, key.fileobj, trainer)

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

          if parentproc != None and not parentproc.is_running():
            parentisdead = True
            log_msg('Found dead parent, exiting')
            if timer is not None:
              timer.cancel()

          if message._serverexception is not None and not message._send_buffer:
            servererror = message._serverexception
            log_msg(servererror)
            ex_code = 1
          if mask & selectors.EVENT_WRITE and not message._send_buffer:
            killreq = message._killreq

          if servererror is not None or killreq or parentisdead or ex_code != 0:
            break
          
          trainer = message._trainer

  except KeyboardInterrupt:
    log_msg('caught keyboard interrupt, exiting')
    ex_code = 130

  except trainlib.ExitCommand:
    log_msg('Shutdown requested, exiting')

  except trainlib.ParentExitCommand:
    log_msg('Found dead parent, exiting')
    ex_code = 2

  except Exception:
    log_msg("Train server error:", traceback.format_exc())
    ex_code = 1

  finally:
    if timer is not None:
      timer.cancel()

    sel.close()

  return ex_code

if __name__ == '__main__':
  sys.exit(main())
