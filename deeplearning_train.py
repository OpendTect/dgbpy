#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : Bert
# DATE     : August 2018
#
# Script for deep learning train
# is called by the DeepLearning plugin
#

from odpy.common import *
import odpy.iopar as iopar

import sys
import os.path
import argparse
import subprocess

parser = argparse.ArgumentParser(prog='PROG',description='Training a machine learning model')
parser.add_argument('parfile',type=argparse.FileType('r'),
                    help='OpendTect parameter file')
parser.add_argument('-v','--version',action='version',version='%(prog)s 1.0')
parser.add_argument('--log',dest='logfile',metavar='file',nargs='?',type=argparse.FileType('a'),
                    default='sys.stdout',help='Progress report output')
parser.add_argument('--syslog',dest='sysout',metavar='stdout',nargs='?',type=argparse.FileType('a'),
                    default='sys.stdout',help='Standard output')
args = vars(parser.parse_args())
initLogging(args)

inpfile = args['parfile'].name
std_msg( "Reading pars from " + inpfile )
inpfile = open( inpfile, "r" )

def read_iopar_line( fp ):
  return iopar.read_line( fp, False )

apply_type = ""
data_file = ""
while True:

  res = read_iopar_line( inpfile )
  ky = res[0]
  if ky == "!":
    break;

  if ky == "Data File":
    data_file = res[1]
  if ky == "Learning Type":
    apply_type = res[1]
  if ky == "Log File":
    set_log_file( res[1] )

log_msg( "Deeplearning Training Module Started" )
std_msg( "Data file is: " + data_file )

if not os.path.exists( data_file ):
  log_msg( "Error: data file does not exist: " + data_file )
  exit( 1 )

guipath = os.path.join( os.path.dirname(os.path.realpath(__file__)), 'keras_gui.py' )
logfilenm = get_log_file()
stdoutfilenm = get_stdlog_file()

subprocess.call(['python3',guipath,data_file,'--log',logfilenm,'--syslog',stdoutfilenm])

log_msg( "Deeplearning Training Module Finished" )
sys.exit( 0 )
