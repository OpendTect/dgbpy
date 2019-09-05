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

from odpy import common as odcommon
from odpy.oscommand import getPythonExecNm, printProcessTime
import dgbpy.mlapply as dgbmlapply

parser = argparse.ArgumentParser(
            description='Run machine learning model training')
parser.add_argument( '-v', '--version',
            action='version', version='%(prog)s 1.0' )
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
datagrp.add_argument( '--mldir',
            dest='mldir', nargs=1,
            help='Machine Learning Directory' )
odappl = parser.add_argument_group( 'OpendTect application' )
odappl.add_argument( '--dtectexec',
            metavar='DIR', nargs=1,
            help='Path to OpendTect executables' )
loggrp = parser.add_argument_group( 'Logging' )
loggrp.add_argument( '--proclog',
            dest='logfile', metavar='file', nargs='?',
            type=argparse.FileType('w'), default=sys.stdout,
            help='Progress report output' )
loggrp.add_argument( '--syslog',
            dest='sysout', metavar='stdout', nargs='?',
            type=argparse.FileType('a'), default=sys.stdout,
            help='Standard output' )
args = vars(parser.parse_args())
odcommon.initLogging( args )
odcommon.proclog_logger.setLevel( 'DEBUG' )

if __name__ == '__main__':
  odcommon.log_msg( 'Starting program:', getPythonExecNm(), " ".join(sys.argv) )
  odcommon.log_msg( 'Processing on:', platform.node() )
  odcommon.log_msg( 'Process ID:', os.getpid(), '\n' )
  printProcessTime( 'Machine Learning Training', True, odcommon.log_msg )
  odcommon.log_msg( '\n' )
  dict = json.loads( args['dict'][0] )
  try:
    success = dgbmlapply.doTrain( args['h5file'].name, dict['platform'],
                                  dict['parameters'], dict['output'], args )
  except Exception as e:
    odcommon.log_msg( 'Exception:', e )
    sys.exit(1)
  if not success:
    sys.exit(1)

  odcommon.log_msg( '\n' )
  printProcessTime( 'Machine Learning Training', False, odcommon.log_msg )
  odcommon.log_msg( "Finished batch processing.\n" )
  sys.exit(0)
