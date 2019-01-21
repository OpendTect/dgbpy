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

import argparse
import numpy
import struct
import sys


# -- IO tools

inpstrm = sys.stdin.buffer
outtxtstrm = sys.stdout
outstrm = outtxtstrm.buffer

def exit_err( msg ):
  outstrm.write( b"ERR" )
  outtxtstrm.write( str(len(msg)) + ' ' + msg + '\n' )
  exit( 1 )


# -- command line parser

parser = argparse.ArgumentParser(
            description='Application of a trained machine learning model')
parser.add_argument( '-v', '--version',
            action='version',version='%(prog)s 1.0')
parser.add_argument( '--log',
            dest='logfile', metavar='file', nargs='?',
            type=argparse.FileType('a'), default='sys.stdout',
            help='Progress report output' )
parser.add_argument( '--syslog',
            dest='sysout', metavar='stdout', nargs='?',
            type=argparse.FileType('a'), default='sys.stdout',
            help='Standard output' )
parser.add_argument( 'parfilename',
            help='The input parameter file' )
args = parser.parse_args()
initLogging( vars(args) )
print( args.parfilename )


# -- read parameter file

try:
  parfile = open( args.parfilename, "r" );
except IOError:
  exit_err( "Cannot open parameter file " + args.parfilename )

def read_iopar_line():
  return iopar.read_line( parfile, False )

zstart = 0
zstop = -1
zstep = 1
keras_file = ""
outputs = []
total_nr_inpsamps = 0
nroutsamps = 0

for i in range(4): # dispose of file header
  read_iopar_line()

while True:

  ioparkeyval = read_iopar_line()
  ky = ioparkeyval[0]
  if ky == "!":
    break;
  val = ioparkeyval[1]

  if ky == "Size":
    total_nr_inpsamps = int( val )
  elif ky == "Z range":
    nrs = val.split( "`" )
    zstart = float( nrs[0] )
    zstop = float( nrs[1] )
    zstep = float( nrs[2] )
  elif ky == "Z length":
    nroutsamps = int( val )
  elif ky == "File name":
    keras_file = val
  elif ky == "Output":
    nrs = val.split( "`" )
    for idx in range( len(nrs) ):
      outputs.append( int(nrs[idx]) )

parfile.close()


# -- sanity checks, initialisation

if nroutsamps < 1:
  exit_err( "did not see 'Size' key in input IOPar" )

nroutvals = len( outputs )
if nroutvals < 1:
  exit_err( "No 'Output's found in par file" )

nrprocessed = 0


# -- operation

while True:

  actcode = inpstrm.read( 3 );
  if len(actcode) < 3:
    exit_err( "Cannot get actcode" )

  actstr = actcode.decode('utf8')
  if actstr == "STP":
    break

  if actstr != "INP":
    exit_err( "Unknown actcode" )

  # read total_nr_inpsamps floats
  try:
    inpdata = inpstrm.read( 4*total_nr_inpsamps )
  except:
    exit_err( "Data transfer failure" )

  vals = struct.unpack( 'f'*total_nr_inpsamps, inpdata )
  outvals = numpy.zeros( shape=(outnr*nroutvals) )

# TODO: -- implement keras apply
  slicesz = int(total_nr_inpsamps) // int(nroutsamps) # incorrect but who cares
  for iout in range( 0, nroutsamps ):
    valwindow = vals[ iout*slicesz : (iout+1)*slicesz ]
    outnr = 0
    def set_outval( val ):
      outvals[outnr*nroutvals + iout] = numpy.std( valwindow )
      ++outnr

    if 0 in outputs:
      set_outval( numpy.mean(valwindow) )
    if 1 in outputs:
      set_outval( numpy.std(valwindow) )
# --

  outstrm.write( "OUT" )
  outstrm.write( struct.pack(outvals) )
  nrprocessed = nrprocessed + 1

os.remove( args.parfilename )
exit( 0 )
