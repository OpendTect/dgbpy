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
import numpy as np
import struct
import sys
import time

import dgbpy.keystr as dgbkeys
import dgbpy.hdf5 as dgbhdf5
import dgbpy.mlio as dgbmlio
import dgbpy.mlapply as dgbmlapply

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
parser.add_argument( '--ascii', dest='binaryout', action='store_false',
                     default=True,
                     help="write ascii text to output buffer" )
parser.add_argument( '--fakeapply', dest='fakeapply', action='store_true',
                     default=False,
                     help="applies a numpy average instead of the model" )
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
binaryout = args.binaryout
fakeapply = args.fakeapply
debug_mode = getattr(args,'debug') or getattr(args,'wait')
if debug_mode:
  from pdb_clone import pdbhandler
  pdbhandler.register()
  if getattr(args,'wait'):
    from pdb_clone import pdb
    pdb.set_trace_remote()

# -- I/O tools

def put_to_output( what, isact=False ):
  if binaryout:
    if isact:
      ret = mk_actioncode_bytes( what )
    else:
      if what.dtype != np.float32:
        what = np.array( what, np.float32 )
      ret = what.tobytes()
  else:
    ret = what

  if binaryout:
    return outstrm.write( ret )
  else:
    if isact:
      return outtxtstrm.write( str(ret)+' ' )
    else:
      pos = outtxtstrm.tell()
      ret.tofile( outtxtstrm, sep=' ' )
      outtxtstrm.write( '\n' )
      return outtxtstrm.tell() - pos

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
nroutsamps = -1
tensor_size = 0
fixedincomingnrvals = 0

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
    fixedincomingnrvals = int( val )
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

fixedsize = nroutsamps >= 0

outputnms = list()
if 0 in outputs:
  outputnms.append( dgbkeys.classvalstr )
if 1 in outputs:
  outputnms.append( dgbkeys.confvalstr )

if fakeapply:
  modelinfo = dgbmlio.getInfo( keras_file )
  modelinfo[dgbkeys.plfdictstr] = dgbkeys.numpyvalstr
  model = None
else:
  (model,modelinfo) = dgbmlio.getModel( keras_file )
applyinfo = dgbmlio.getApplyInfo( modelinfo, outputnms )
nrattribs = dgbhdf5.get_nr_attribs( modelinfo )
stepout = modelinfo[dgbkeys.stepoutdictstr]

if fixedsize:
  inp_shape = (nrattribs,2*stepout[0]+1,2*stepout[1]+1,nroutsamps+2*stepout[2])
  examples_shape = dgbhdf5.get_np_shape( stepout, nrattribs=nrattribs,
                                         nrpts=nroutsamps )
  nrz = examples_shape[-1]
  examples = np.empty( examples_shape, dtype=np.float32 )
  outgoingnrvals = nroutputs * nroutsamps
  outgoingnrbytes = outgoingnrvals * 4
else:
  stepout = [0,0,stepout]


# -- sanity checks, initialisation

nroutputs = len( outputs )
if nroutputs < 1:
  exit_err( "No 'Output's found in par file" )

dbg_pr( "At", "1" )


# -- operation

nrprocessed = 0
nrbytes = 0
rdnrvals = 0

start = time.clock()
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

  if fixedsize and rdnrvals != fixedincomingnrvals:
    if nrprocessed == 0:
      exit_err( "Bad nr input samples: " + str(rdnrvals)
                 + " should be " + str(fixedincomingnrvals) )
    break # happens at EOF, too, does not except but gives wild value

  # slurp input for one trace
  try:
    inpdata = get_from_input( 4*rdnrvals )
  except:
    exit_err( "Data transfer failure" )

  if not fixedsize:
    nroutsamps = int(rdnrvals/nrattribs) - 2*stepout[2]
    inp_shape = (nrattribs,2*stepout[0]+1,2*stepout[1]+1,nroutsamps+2*stepout[2])
    examples_shape = dgbhdf5.get_np_shape( stepout, nrattribs=nrattribs,
                                           nrpts=nroutsamps )
    nrz = examples_shape[-1]
    examples = np.empty( examples_shape, dtype=np.float32 )
    outgoingnrvals = nroutputs * nroutsamps
    outgoingnrbytes = outgoingnrvals * 4


  valsret = np.reshape( np.frombuffer(inpdata,dtype=np.float32), inp_shape )
  for zidz in range(nroutsamps):
    examples[zidz] = valsret[:,:,:,zidz:zidz+nrz]

  ret = dgbmlapply.doApply( model, modelinfo, examples, applyinfo )
  outvals = ret[dgbkeys.preddictstr]
# --

  # success ... write nr values and the trace/log data
  put_to_output( outgoingnrvals, isact=True )
  nrbyteswritten = put_to_output( outvals )
  if binaryout and nrbyteswritten != outgoingnrbytes:
    exit_err( "Could only write " + str(nrbyteswritten)
              + " of " + str(outgoingnrbytes) )
  nrprocessed = nrprocessed + 1

duration = time.clock()-start
std_msg( "Total time:",  "{:.3f}".format(duration), "s.;", \
         "{:.3f}".format(nrprocessed/duration), "tr/s." )

# for production, uncomment to keep /tmp tidy
#os.remove( parfile.name )
exit( 0 )
