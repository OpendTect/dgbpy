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
# The protocol is to simply transmit the number of (float32) values and then the data,
# both ways.
# From input we get a full Z range of data, i.e. the entire input for one trace or log.
# Back, we deliver the output trace/log, optionally also one or more aux traces/logs.
#

from odpy.common import *
import odpy.iopar as iopar

import argparse
import numpy
import struct
import sys

def mk_actioncode_bytes( ival ):
  return ival.to_bytes( 4, byteorder=sys.byteorder, signed=True )

def exit_err( msg ):
  errcode = -1
  outstrm.write( mk_actioncode_bytes(errcode) )
  outtxtstrm.write( str(len(msg)) + ' ' + msg + '\n' )
  outtxtstrm.flush()
  exit( 1 )

dbg_strm = open( "/tmp/da_dbg.txt", "w" )
def dbg_pr( what, val ):
  dbg_strm.write( what + ": " + val + "\n" )
  dbg_strm.flush()

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

# -- I/O tools

def put_to_output( what ):
  return outstrm.write( what )

def get_from_input( nrbytes ):
  return inpstrm.read( nrbytes )

def mk_int_bytes( ival ):
  return ival.to_bytes( 4, byteorder=sys.byteorder, signed=True )

def get_int_from_bytes( data_read ):
  return int.from_bytes( data_read, byteorder=sys.byteorder, signed=True )


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
    break
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

dbg_pr( "At", "1" )


# -- operation

nrprocessed = 0
nrbytes = 0
rdnrvals = 0
outgoingnrvals = nroutputs * nroutsamps
outgoingnrbytes = outgoingnrvals * 4

while True:

  # read action code
  try:
    data_read = get_from_input( 4 )
  except:
    break

  # anything positive should be the number of values
  rdnrvals = get_int_from_bytes( data_read )
  dbg_pr( "At", "2" )

  if rdnrvals < 0:
    break
  if rdnrvals == 0:
    time.sleep( 0.01 ) # avoids crazy CPU usage
    continue
  dbg_pr( "At", "3" )

  if rdnrvals != incomingnrvals:
    if nrprocessed == 0:
      exit_err( "Bad nr input samples: " + str(rdnrvals)
                 + " should be " + str(incomingnrvals) )
    break # happens at EOF, too, does not except but gives wild value

  # slurp input for one trace
  try:
    inpdata = get_from_input( 4*incomingnrvals )
  except:
    exit_err( "Data transfer failure" )

  vals = struct.unpack( 'f'*incomingnrvals, inpdata )
  outvals = numpy.zeros( outgoingnrvals, dtype=numpy.float32 )

# TODO: -- implement keras apply
  # the following is just to return something
  slicesz = incomingnrvals // nroutsamps
  for iout in range( 0, nroutsamps ):
    valwindow = vals[ iout*slicesz : (iout+1)*slicesz ]
    outnr = 0
    def set_outval( val ):
      outvals[outnr*nroutsamps + iout] = val
      ++outnr

    if 0 in outputs:
      set_outval( numpy.mean(valwindow) )
    if 1 in outputs:
      set_outval( numpy.std(valwindow) )
# --

  # success ... write nr values and the trace/log data
  put_to_output( mk_actioncode_bytes(outgoingnrvals) )
  nrbyteswritten = put_to_output( outvals.tobytes() )
  if nrbyteswritten != outgoingnrbytes:
    exit_err( "Could only write " + str(nrbyteswritten)
              + " of " + str(outgoingnrbytes) )
  nrprocessed = nrprocessed + 1

# for production, uncomment to keep /tmp tidy
#os.remove( parfile.name )
exit( 0 )
