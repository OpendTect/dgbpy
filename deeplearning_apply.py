#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : Bert
# DATE     : July 2018
#
# Script for deep learning apply
# is called by the DeepLearning plugin
#
# Required: par file as created by OD's deeplearning module
# Normal operation is receiving data from stdin and putting results on stdout
# The protocol is a simple 3-char code before sending binary float32 data,
# both ways.
#

from odpy.common import *
import odpy.iopar as iopar

import argparse
import numpy
import struct
import sys

def get_actioncode_bytes( ival ):
  return ival.to_bytes( 4, byteorder=sys.byteorder, signed=True )

def exit_err( msg ):
  errcode = -1
  outstrm.write( get_actioncode_bytes(errcode.to_bytes) )
  outtxtstrm.write( str(len(msg)) + ' ' + msg + '\n' )
  outtxtstrm.flush()
  exit( 1 )

# -- command line parser

parser = argparse.ArgumentParser(
            description='Application of a trained machine learning model')

# standard
parser.add_argument( '-v', '--version',
            action='version',version='%(prog)s 1.0')
parser.add_argument( '--log',
            dest='logfile', metavar='file', nargs='?',
            type=argparse.FileType('a'), default=sys.stdout,
            help='Progress report output' )
parser.add_argument( '--syslog',
            dest='sysout', metavar='stdout', nargs='?',
            type=argparse.FileType('a'), default=sys.stdout,
            help='System log' )

# optional
parser.add_argument( '-i','--input', nargs='?', type=argparse.FileType('r'),
                     default=sys.stdin )
parser.add_argument( '-o','--output', nargs='?', type=argparse.FileType('w'),
                     default=sys.stdout )
parser.add_argument( '--debug', help="prepare for pdb-clone debugging",
                      action="store_true")
parser.add_argument( '--wait', help="wait execution for pdb-clone debugger to attach",
                      action="store_true")

# required
parser.add_argument( 'parfilename', type=argparse.FileType('r'),
                     help='The input parameter file' )

args = parser.parse_args()
vargs = vars( args )
initLogging( vargs )

parfile = args.parfilename
inpstrm = args.input.buffer
outtxtstrm = args.output
outstrm = outtxtstrm.buffer
debug_mode = getattr(args,'debug') or getattr(args,'wait')
if debug_mode:
  from pdb_clone import pdbhandler
  pdbhandler.register()
  if getattr(args,'wait'):
    from pdb_clone import pdb
    pdb.set_trace_remote()

# -- read parameter file

def read_iopar_line():
  return iopar.read_line( parfile, False )

zstart = 0
zstop = -1
zstep = 1
keras_file = ""
outputs = []
nroutsamps = 0
tensor_size = 0
incomingnrvals = 0

for i in range(4): # dispose of file header
  read_iopar_line()

while True:

  ioparkeyval = read_iopar_line()
  ky = ioparkeyval[0]
  if ky == "!":
    break;
  val = ioparkeyval[1]

  if ky == "File name":
    keras_file = val
  elif ky == "Input.Size":
    incomingnrvals = int( val )
  elif ky == "Input.Size.Z":
    nroutsamps = int( val )
  elif ky == "Tensor.Size":
    tensor_size = int( val )
  elif ky == "Z range":
    nrs = val.split( "`" )
    zstart = float( nrs[0] )
    zstop = float( nrs[1] )
    zstep = float( nrs[2] )
  elif ky == "Output":
    nrs = val.split( "`" )
    for idx in range( len(nrs) ):
      outputs.append( int(nrs[idx]) )

parfile.close()


# -- sanity checks, initialisation

if nroutsamps < 1:
  exit_err( "did not see 'Input.Size.Z' key in input IOPar" )

nroutputs = len( outputs )
if nroutputs < 1:
  exit_err( "No 'Output's found in par file" )

nrprocessed = 0


# -- operation

nrbytes = 0
rdnrvals = 0
outgoingnrvals = nroutputs * nroutsamps

while True:

  rdnrvals = int.from_bytes( inpstrm.read(4), byteorder=sys.byteorder );
  if rdnrvals == -1:
    break
  if rdnrvals != incomingnrvals:
    exit_err( "Bad nr input samples: " + str(rdnrvals)
               + " should be " + str(incomingnrvals) )

  # read input for one trace
  try:
    inpdata = inpstrm.read( 4*incomingnrvals )
  except:
    exit_err( "Data transfer failure" )

  vals = struct.unpack( 'f'*incomingnrvals, inpdata )
  outvals = numpy.zeros( shape=(outgoingnrvals), dtype=numpy.float32 )

# TODO: -- implement keras apply
  # the following is just to return something
  slicesz = incomingnrvals // nroutsamps
  for iout in range( 0, nroutsamps ):
    valwindow = vals[ iout*slicesz : (iout+1)*slicesz ]
    outnr = 0
    def set_outval( val ):
      outvals[outnr*nroutsamps + iout] = numpy.std( valwindow )
      ++outnr

    if 0 in outputs:
      set_outval( numpy.mean(valwindow) )
    if 1 in outputs:
      set_outval( numpy.std(valwindow) )
# --

  outstrm.write( get_actioncode_bytes(outgoingnrvals) )
  nrbytes = outstrm.write( outvals.astype('float32') )
  nrprocessed = nrprocessed + 1

# for production, uncomment to keep /tmp tidy
#os.remove( parfile.name )
exit( 0 )
