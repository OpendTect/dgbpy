#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : Bert
# DATE     : August 2018
#
# Script for deep learning train
# is called by the DeepLearning plugin
#

import odpy.iopar
import sys
import os.path

inpfile = sys.stdin
outfile = sys.stdout

def dbg_msg( s ):
  odpy.common.dbg_msg( s )

dbg_msg( "Entered deeplearning_train" )

if len(sys.argv) < 2:
  dbg_msg( "Reading pars from stdin" )
else:
  dbg_msg( "Reading pars from " + sys.argv[1] )
  inpfile = open( sys.argv[1], "r" )

def read_iopar_line( fp ):
  return odpy.iopar.read_line( fp, False )

apply_type = ""
data_file = ""
standalone = 0
while True:

  res = read_iopar_line( inpfile )
  ky = res[0]
  if ky == "!":
    break;

  if ky == "Type":
    apply_type = res[1]
  if ky == "File name":
    data_file = res[1]
  if ky == "Standalone":
    standalone = res[1].lower()[:1] == "y"

dbg_msg( "Data file is: " + data_file )

if not os.path.exists( data_file ):
  dbg_msg( "Error: data file does not exist: " + data_file )
  exit( 1 )

from subprocess import call
call(["od_DispMsg", "Training not implemented yet"])

dbg_msg( "Quitting deeplearning_train" )
exit( 0 )
