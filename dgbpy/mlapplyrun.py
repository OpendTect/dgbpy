#!/usr/bin/env python3
#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Mar 2019
#
# _________________________________________________________________________
# runs a machine learning training a stand-alone process
#

import sys
import os
import platform
import argparse
import json
import traceback as tb

from odpy.common import initLogging, isLux, log_msg
from odpy.oscommand import getPythonExecNm, printProcessTime
import dgbpy.keystr as dgbkeys
from dgbpy.mlio import load_libstdcpp
import dgbpy.mlapply as dgbmlapply

from odpy import common as odcommon
odcommon.proclog_logger.setLevel( 'DEBUG' )

def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
              description='Run machine learning model training')
  parser.add_argument( '-v', '--version',
              action='version', version='%(prog)s 1.1' )
  parser.add_argument( 'h5file',
              type=argparse.FileType('r'),
              help='HDF5 file containing the training data' )
  parser.add_argument( '--dict',
              dest='dict', metavar='JSON_DICT', nargs=1,
              help='Dictionary: {"platform": "keras", "output": "<new model>", "parameters": {trainpars}}' )
  datagrp = parser.add_argument_group( 'Data' )
  datagrp.add_argument( '--dataroot',
              dest='dtectdata', metavar='DIR', nargs=1,
              help='Survey Data Root' )
  datagrp.add_argument( '--survey',
              dest='survey', nargs=1,
              help='Survey name' )
  odappl = parser.add_argument_group( 'OpendTect application' )
  odappl.add_argument( '--dtectexec',
              metavar='DIR', nargs=1,
              help='Path to OpendTect executables' )
  odappl.add_argument( '--odpid',
              dest='odpid', action='store',
              type=int, default=-1,
              help='Process ID of the OpendTect client application' )
  loggrp = parser.add_argument_group( 'Logging' )
  loggrp.add_argument( '--proclog',
              dest='logfile', metavar='file', nargs='?',
              type=argparse.FileType('w'), default=sys.stdout,
              help='Progress report output' )
  loggrp.add_argument( '--syslog',
              dest='sysout', metavar='stdout', nargs='?',
              type=argparse.FileType('a'), default=sys.stdout,
              help='Standard output' )
  return parser.parse_args()

def main() -> int:
  args = vars(parse_args())

  initLogging( args )

  log_msg( 'Starting program:', getPythonExecNm(), " ".join(sys.argv) )
  log_msg( 'Processing on:', platform.node() )
  log_msg( 'Process ID:', os.getpid(), '\n' )
  printProcessTime( 'Machine Learning Training', True, print_fn=log_msg )
  log_msg( '\n' )

  if isLux() and 'odpid' in args:
    try:
      load_libstdcpp( args['odpid'] )
    except Exception as e:
      log_msg( repr(e) )
      pass

  dict = json.loads( args['dict'][0] )
  traintype = None
  if dgbkeys.learntypedictstr in dict:
    traintype = dgbmlapply.TrainType[ dict[dgbkeys.learntypedictstr] ]
  model = dict.get('model', None)
  logdir = dict.get('logdir', None)
  clearlogs = dict.get('cleanlogdir', False)
  try:
    success = dgbmlapply.doTrain( args['h5file'].name,
                                  platform=dict['platform'],
                                  type=traintype,
                                  params=dict['parameters'],
                                  logdir=logdir,
                                  clearlogs=clearlogs,
                                  modelin=model,
                                  outnm=dict['output'],
                                  bokeh = True,
                                  args=args )
  except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    stackstr = ''.join(tb.extract_tb(exc_tb,limit=20).format())
    log_msg( 'Training error exception:' )
    log_msg( repr(e), 'on line', exc_tb.tb_lineno, 'of script', fname )
    log_msg( stackstr )
    return 1
  if not success:
    return 1

  log_msg( '\n' )
  printProcessTime( 'Machine Learning Training', False, print_fn=log_msg )
  log_msg( "Finished batch processing.\n" )
  return 0

if __name__ == '__main__':
  sys.exit(main())
