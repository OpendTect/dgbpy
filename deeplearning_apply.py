#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : Bert
# DATE     : July 2018
#
# Script for deep learning apply
# is called by the DeepLearning plugin
#

from odpy.common import *
import odpy.iopar as iopar

import sys
import struct
import numpy

#inpfile = open( "/tmp/inp.dat", "rb" );
# for binary read, need to use stdin.buffer
inpfile = sys.stdin.buffer
#outfile = open( "/tmp/out.dat", "w" );
outfile = sys.stdout
lm = LogManager(argv)

def send_msg( typ, msg ):
  print( typ + ": " + msg, file=outfile )

def read_iopar_line( inpfile ):
  return iopar.read_line( inpfile, True )

lm.log_msg( "Deeplearning Apply Module Started" )

arrsz = 0
nn_file = ""
apply_type = ""
while True:

  res = read_iopar_line( inpfile )
  ky = res[0]
  if ky == "!":
    break;

  if ky == "Type":
    apply_type = res[1]
  if ky == "Data Size":
    arrsz = int(res[1])
  if ky == "File name":
    nn_file = res[1]

# TODO read nn_file here
nn_arrsz = arrsz

if nn_arrsz < 1:
  send_msg( "ERR", "Cannot read deep learning network:" + nn_file )

# sanity check
if arrsz > 0 and arrsz != nn_arrsz:
  send_msg( "ERR", "Input size mismatch (read "
                 + str(nn_arrsz) + ", got" + str(arrsz) + ")" )
  exit( 1 )

# report agreed number of floats per input tensor that needs to be transferred
send_msg( "RDY", str(nn_arrsz) )
nrprocessed = 0

# to reduce the data transfer duplication, a 'sliding' version of the input
# arrays can be transferred.
# we need to 'window' through the data to get the actual apply tensors

while True:

  rawnrpts = inpfile.read( 4 );
  if len(rawnrpts) < 4 or rawnrpts.decode('utf8') == "STOP":
    break;
  nrptsarr = struct.unpack( 'i', rawnrpts )
  nrpts = nrptsarr[0]
  if nrpts < 1:
    break
  if nrpts > 1000000:
    send_msg( "ERR", "nrpts=" + str(nrpts)
                     + "should be < 1000000 (sanity check)" )
    break

  totsz = nn_arrsz + nrpts - 1
  vals = struct.unpack( 'f'*totsz, inpfile.read(4*totsz) )
  for idx in range( 0, nrpts ):

#  TODO apply NN here
#  Expecting 2 numbers on output: prediction and confidence

    valwindow = vals[idx:idx+nn_arrsz]
    outstr = str(numpy.mean(valwindow)) + " " + str(numpy.std(valwindow))
    send_msg( "RES", outstr )
    nrprocessed = nrprocessed + 1


send_msg( "BYE", str(nrprocessed) )
lm.std_msg( "deeplearning_apply exiting" )
exit( 0 )
