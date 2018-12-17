#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : Bert
# DATE     : August 2018
#
# Script for deep learning train
# is called by the DeepLearning plugin
#

from odpy.common import *

import sys
import os.path
import argparse
from subprocess import call

parser = argparse.ArgumentParser(prog='PROG',description='Training a machine learning model')
parser.add_argument('train',type=argparse.FileType('r'),help='Training dataset')
parser.add_argument('-v','--version',action='version',version='%(prog)s 1.0')
datagrp = parser.add_argument_group('Data')
datagrp.add_argument('--dataroot',dest='dtectdata',metavar='DIR',nargs=1,
                     help='Survey Data Root')
datagrp.add_argument('--survey',dest='survey',nargs=1,
                     help='Survey name')
odappl = parser.add_argument_group('OpendTect application')
odappl.add_argument('--dtectexec',metavar='DIR',nargs=1,help='Path to OpendTect executables')
odappl.add_argument('--qtstylesheet',metavar='qss',nargs=1,type=argparse.FileType('r'),
                    help='Qt StyleSheet template')
loggrp = parser.add_argument_group('Logging')
loggrp.add_argument('--log',dest='logfile',metavar='file',nargs='?',type=argparse.FileType('a'),
                    default='sys.stdout',help='Progress report output')
loggrp.add_argument('--syslog',dest='sysout',metavar='stdout',nargs='?',type=argparse.FileType('a'),
                    default='sys.stdout',help='Standard output')
args = vars(parser.parse_args())
initLogging(args)
log_msg( "Deeplearning Training Module Started" )
log_msg( " " )

guipath = os.path.join( os.path.dirname(os.path.realpath(__file__)), 'keras_gui.py' )

machcmd = list()
machcmd.append( 'python3' )
machcmd.append( guipath )
machcmd.append( args['train'].name )
dataroot = args['dtectdata']
if dataroot != None:
  machcmd.append( '--dataroot' )
  machcmd.append( dataroot[0] )
survey = args['survey']
if survey != None:
  machcmd.append( '--survey' )
  machcmd.append( survey[0] )
dtectexec = args['dtectexec']
if dtectexec != None:
  machcmd.append( '--dtectexec' )
  machcmd.append( dtectexec[0] )
stylesheet = args['qtstylesheet']
if stylesheet != None:
  machcmd.append( '--qtstylesheet' )
  machcmd.append( stylesheet[0].name )
if 'logfile' in args:
  machcmd.append( '--log' )
  machcmd.append( args['logfile'].name )
if 'syslog' in args:
  machcmd.append( '--syslog' )
  machcmd.append( args['sysout'] )

try:
  retcode = call( machcmd, stdout=args['logfile'], stderr=args['sysout'] )
  if retcode < 0:
    std_msg("Terminated by signal", -retcode)
except OSError as e:
  log_msg("Execution failed:", e)
  raise

log_msg( " " )
log_msg( "Deeplearning Training Module Finished" )
log_msg( " " )
log_msg( "Finished batch processing." )
sys.exit( 0 )
