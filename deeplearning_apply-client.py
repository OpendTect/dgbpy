#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : A. Huck
# DATE     : June 2019
#
# Deep learning apply server
#
#

import argparse
import numpy as np
from os import path
import selectors
import socket
import sys
import time
import traceback

from odpy.common import *
from odpy import oscommand
import dgbpy.keystr as dgbkeys
import dgbpy.hdf5 as dgbhdf5
import dgbpy.deeplearning_apply_clientlib as applylib

sel = selectors.DefaultSelector()

# -- command line parser

parser = argparse.ArgumentParser(
          description='Client application of a trained machine learning model')
parser.add_argument( '-v', '--version',
            action='version',version='%(prog)s 2.0')
datagrp = parser.add_argument_group( 'Data' )
datagrp.add_argument( 'modelfile', type=argparse.FileType('r'),
                       help='The input trained model file' )
datagrp.add_argument( '--examplefile', 
                       dest='examples', metavar='file', nargs='?',
                       type=argparse.FileType('r'),
                       help='Examples file to be applied' )
netgrp = parser.add_argument_group( 'Network' )
netgrp.add_argument( '--address',
            dest='addr', metavar='ADDRESS', action='store',
            type=str, default='localhost',
            help='Address to listen on' )
netgrp.add_argument( '--port',
            dest='port', action='store',
            type=int, default=65432,
            help='Port to listen on')
loggrp = parser.add_argument_group( 'Logging' )
loggrp.add_argument( '--log',
            dest='logfile', metavar='file', nargs='?',
            type=argparse.FileType('w'), default=sys.stdout,
            help='Progress report output' )
loggrp.add_argument( '--syslog',
            dest='sysout', metavar='stdout', nargs='?',
            type=argparse.FileType('w'), default=sys.stdout,
            help='System log' )
loggrp.add_argument( '--server-log',
            dest='servlogfile', metavar='file', nargs='?',
            type=argparse.FileType('w'), default=sys.stdout,
            help='Python server log' )
loggrp.add_argument( '--server-syslog',
            dest='servsysout', metavar='stdout', nargs='?',
            type=argparse.FileType('w'), default=sys.stdout,
            help='Python server System log' )
# optional
parser.add_argument( '--fakeapply', dest='fakeapply', action='store_true',
                     default=False,
                     help="applies a numpy average instead of the model" )
parser.add_argument( '--local', dest='localserv', action='store_true',
                     default=False,
                     help="use a local network socket connection" )


args = vars(parser.parse_args())
initLogging( args )
modelfnm = args['modelfile'].name
local = args['localserv']

servscriptfp =  path.join(path.dirname(__file__),'deeplearning_apply-server.py')
servercmd = list()
servercmd.append( oscommand.getPythonExecNm() )
servercmd.append( servscriptfp )
servercmd.append( modelfnm )
servercmd.append( '--address' )
servercmd.append( str(args['addr']) )
servercmd.append( '--port' )
servercmd.append( str(args['port']) )
if args['servlogfile'].name != '<stdout>':
  servercmd.append( '--log' )
  servercmd.append( args['servlogfile'].name )
if args['servsysout'].name != '<stdout>':
  servercmd.append( '--syslog' )
  servercmd.append( args['servsysout'].name )
if args['fakeapply']:
  servercmd.append( '--fakeapply' )
if local:
  servercmd.append( '--local' )

serverproc = oscommand.execCommand( servercmd, True )
time.sleep( 2 )

def getApplyTrace( dict ):
  arr3d = dict['arr']
  shape = dict['inp_shape']
  procstep = dict['step']-1
  idx = dict['idx']
  idy = dict['idy']
  return arr3d[:,idx:idx+shape[0],\
                 idy:idy+shape[1]+procstep+1,:]

def create_request(action, value=None):
  if value == None:
    return dict(
      type="text/json",
      encoding="utf-8",
      content=dict(action=action),
    )
  elif action == 'outputs':
    return dict(
      type="text/json",
      encoding="utf-8",
      content=dict(
        action=action,
        value= {
          'names': value,
          dgbkeys.surveydictstr: 'None',
          dgbkeys.dtypepred: 'uint8',
          dgbkeys.dtypeprob: 'float32',
          dgbkeys.dtypeconf: 'float32'
        },
      ),
    )
  elif action == 'data':
    arr = getApplyTrace(value)
    return dict(
      type='binary/array',
      encoding=[arr.dtype.name],
      content=[arr],
    )
  else:
    return dict(
        type="binary/custom-client-binary-type",
        encoding="binary",
        content=bytes(action + value, encoding="utf-8"),
    )

def req_connection(host, port, request):
  if local:
    addr = str(port)
    sockfam = socket.AF_UNIX
  else:
    addr = (host, port)
    sockfam = socket.AF_INET
  sock = socket.socket(sockfam, socket.SOCK_STREAM)
  sock.setblocking(True)
  sock.connect_ex(addr)
  events = selectors.EVENT_READ | selectors.EVENT_WRITE
  message = applylib.Message(sel, sock, addr, request)
  sel.register(sock, events, data=message)

def getApplyPars( args ):
  if args['examples'] == None:
    ret= {
      'inp_shape': [33,33,33],
      'nrattribs': 1,
      'outputnms': dgbhdf5.getOutputNames(modelfnm,[0]),
      'surveydirnm': 'None'
    }
  else:
    exfnm = args['examples'].name
    info = dgbhdf5.getInfo( exfnm )
    shape = info['inp_shape']
    if not isinstance(shape,list):
      shape=[0,0,shape]
    ret = {
      'inp_shape': shape,
      'nrattribs': dgbhdf5.getNrAttribs( info ),
      'outputnms': dgbhdf5.getOutputs( exfnm )
    }
  return ret

pars = getApplyPars( args )
shape = pars['inp_shape']

nrattribs = pars['nrattribs']
nrlines_out = 20
nrtrcs_out = 800
chunk_step = 50
nrlines_in = nrlines_out + shape[0] - 1
nrtrcs_in = nrtrcs_out + shape[1] - 1
nrz_in = 378
nrz_out = nrz_in - shape[2] + 1
inpdata = np.random.random( nrattribs*nrlines_in*nrtrcs_in*nrz_in )
inpdata = inpdata.reshape((nrattribs,nrlines_in,nrtrcs_in,nrz_in))
inpdata = inpdata.astype('float32')

start = time.time()

host,port = args['addr'], args['port']
req_connection(host, port, create_request('status'))
req_connection(host, port, create_request('outputs',pars['outputnms']))
applydict = {
  'arr': inpdata,
  'inp_shape': shape,
  'idx': int(shape[0]/2-1),
  'idy': int(shape[1]/2-1),
  'step': chunk_step,
}
lastidy = nrtrcs_in-int(shape[1]/2-1)
nrrepeats = 1
trcrg = range(0,nrtrcs_in-shape[1]+1)
nrtrcs = (nrrepeats * len(trcrg))
for i in range(nrrepeats):
  applydict['idx'] = i
  for idy in range(0,nrtrcs_in-shape[1]+1,chunk_step):
    applydict['idy'] = idy
    req_connection(host, port, create_request('data',applydict))

req_connection(host, port, create_request('kill'))

try:
  while True:
    events = sel.select(timeout=1)
    for key, mask in events:
      message = key.data
      try:
        message.process_events(mask)
      except Exception:
        std_msg(
            "main: error: exception for",
            f"{message.addr}:\n{traceback.format_exc()}",
        )
        message.close()
    # Check for a socket being monitored to continue.
    if not sel.get_map():
      break
except KeyboardInterrupt:
  std_msg("caught keyboard interrupt, exiting")
finally:
  sel.close()
  oscommand.kill( serverproc )
  duration = time.time()-start
  log_msg( "Total time:",  "{:.3f}".format(duration), "s.;", \
         "{:.3f}".format(nrtrcs/duration), "tr/s." )
