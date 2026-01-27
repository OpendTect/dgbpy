#!/usr/bin/env python3
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
import psutil
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


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
            description='Client application of a trained machine learning model')
  parser.add_argument( '-v', '--version',
                       action='version',version='%(prog)s 3.0' )
  datagrp = parser.add_argument_group( 'Data' )
  datagrp.add_argument( 'modelfile', type=argparse.FileType('r'),
                         help='The input trained model file' )
  parsgrp = parser.add_argument_group( 'Global inference parameters' )
  parsgrp.add_argument( '--2d', dest='data_is2d', action='store_true',
                       default=False,
                       help='Apply using generated 2d data (instead of 3d)')
  parsgrp.add_argument( '--batchsize',
                       dest='batchsize', metavar='SIZE',
                       type=int, default=None,
                       help='The batch size when applying the inference')
  parsgrp.add_argument( '--device',
                       dest='device', metavar='DEVICE',
                       type=str, default='gpu',
                       help='The device used to apply the inference')
  imggrp = parser.add_argument_group( 'Image-to-image group' )
  imggrp.add_argument( '--infersize',
                       dest='infer_size', metavar='SZ',
                       type=int, nargs='+', default=None,
                       help='The inference size if supported by the model' )
  imggrp.add_argument( '--nrimages',
                       dest='nr_images', metavar='NR',
                       type=int, default=8,
                       help='The number of images per batch for applying the inference')
  imggrp.add_argument( '--applydir',
                       dest='apply_dir', metavar='In-line|Cross-line|Average|Minimum|Maximum',
                       type=str, default=None,
                       help='Apply direction (only when applying 2D models on 3D data)')
  pointgrp = parser.add_argument_group( 'Image-to-point group' )
  pointgrp.add_argument( '--inlstep',
                         dest='inlstep', metavar='STEP',
                         type=int, default=5,
                         help='The step between inlines for applying the inference')
  pointgrp.add_argument( '--crlstep',
                         dest='crlstep', metavar='STEP',
                         type=int, default=50,
                         help='The step between crosslines for applying the inference')
  parsgrp.add_argument( '--fakeapply', dest='fakeapply', action='store_true',
                        default=False,
                        help='applies a numpy average instead of the model' )
  netgrp = parser.add_argument_group( 'Network' )
  netgrp.add_argument( '--address',
              dest='addr', metavar='ADDRESS', action='store',
              type=str, default='localhost',
              help='Address to listen on (Use "LOCAL" for a local socket)' )
  netgrp.add_argument( '--port',
              dest='port', action='store',
              type=int, default=None,
              help='Port to listen on')
  loggrp = parser.add_argument_group( 'Logging' )
  loggrp.add_argument( '--log',
              dest='logfile', metavar='file', nargs='?',
              type=argparse.FileType('w'), default=sys.stdout,
              help='Progress report output' )
  loggrp.add_argument( '--syslog',
              dest='sysout', metavar='file', nargs='?',
              type=argparse.FileType('w'), default=sys.stdout,
              help='System log' )
  loggrp.add_argument( '--server-log',
              dest='servlogfile', metavar='file', nargs='?',
              type=str, default='stdout',
              help='Python server log' )
  loggrp.add_argument( '--server-syslog',
              dest='servsysout', metavar='file', nargs='?',
              type=str, default='stdout',
              help='Python server System log' )
  return parser.parse_args()

def get_client_apply_pars( args ):
  modelfnm = args['modelfile'].name
  info = dgbhdf5.getInfo( modelfnm, True )
  platform = info[dgbkeys.plfdictstr]
  img2img = dgbhdf5.isImg2Img( info )
  nrattribs = dgbhdf5.getNrAttribs( info )
  if args['infer_size'] is None:
    shape = info[dgbkeys.inpshapedictstr]
    if not isinstance(shape,list):
      shape=[0,0,shape]
  else:
    shape = args['infer_size']

  ret = {
    dgbkeys.inpshapedictstr: shape,
    dgbkeys.plfdictstr: platform,
    'outputnms': dgbhdf5.getMainOutputs( info ),
    'img2img': img2img
  }

  inpdatashp = None
  batchsize = args['batchsize']
  if img2img:
    nrimages = args['nr_images']
    if batchsize is None:
      batchsize = 1

    data_is2d = args['data_is2d']
    if not data_is2d and dgbhdf5.is2DModel( info ):
      dir = dgbkeys.inlinestr
      if shape[0] == 1 and shape[1] > 1:
        dir = dgbkeys.inlinestr
      elif shape[0] > 1 and shape[1] == 1:
        dir = dgbkeys.crosslinestr

      if args['apply_dir'] is not None:
        dir = args['apply_dir']
        assert dir in (dgbkeys.inlinestr,dgbkeys.crosslinestr,\
                       dgbkeys.averagestr,dgbkeys.minstr,dgbkeys.maxstr),\
                       f'Invalid apply direction: {dir}'
      if dir == dgbkeys.inlinestr and shape[0] > 1 and shape[1] == 1:
        shape = shape[1], shape[0], shape[2]
      elif dir == dgbkeys.crosslinestr and shape[0] == 1 and shape[1] > 1:
        shape = shape[1], shape[0], shape[2]
      elif dir in (dgbkeys.averagestr,dgbkeys.minstr,dgbkeys.maxstr):
        sz = max(shape[0],shape[1])
        shape = sz, sz, shape[2]
      ret['apply_dir'] = dir
      ret[dgbkeys.inpshapedictstr] = shape

    inpdatashp = batchsize * nrimages, nrattribs, *shape
  else:
    nrlines_out = 21
    nrtrcs_out = 1001
    nrz_out = 2000 if dgbhdf5.isLogInput(info) else 376
    ret['outpdata_size'] = (nrlines_out,nrtrcs_out,nrz_out)
    nrlines_in = nrlines_out + (shape[0] - 1 if shape[0]>1 else 0)
    nrtrcs_in = nrtrcs_out + (shape[1] - 1 if shape[1]>1 else 0)
    nrz_in = nrz_out + (shape[2] - 1 if shape[2]>1 else 0)
    inpdatashp = nrattribs, nrlines_in, nrtrcs_in, nrz_in
    if batchsize is None:
      batchsize = 512

  ret['inpdata_size'] = inpdatashp
  ret['batchsize'] = batchsize
  return ret

def get_server_cmd( args, port ):
  servscriptfp =  path.join(path.dirname(__file__),'deeplearning_apply-server.py')
  servercmd = list()
  servercmd.append( oscommand.getPythonExecNm() )
  servercmd.append( servscriptfp )
  servercmd.append( args['modelfile'].name )
  servercmd.append( '--address' )
  servercmd.append( str(args['addr']) )
  servercmd.append( '--port' )
  servercmd.append( str(port) )
  servercmd.append( '--ppid' )
  servercmd.append( str(psutil.Process().pid) )
  if args['servlogfile'] != 'stdout':
    servercmd.append( '--log' )
    servercmd.append( args['servlogfile'] )
  if args['servsysout'] != 'stdout':
    servercmd.append( '--syslog' )
    servercmd.append( args['servsysout'] )
  return servercmd

def get_server_pars( args, applypars, stddev ):
  modelinfo = dgbhdf5.getInfo( args['modelfile'].name, True )
  serverpars = {
    'data_is2d': args['data_is2d'],
    'targetnames': applypars['outputnms'],
    'defaultbatchsz': applypars['batchsize'],
    dgbkeys.prefercpustr: args['device'] == 'cpu',
    dgbkeys.surveydictstr: 'Fake numpy survey'
  }

  if dgbhdf5.applyGlobalStd( modelinfo ):
    serverpars['scales'] = [{
      dgbkeys.dbkeydictstr: '100010.999987',
      dgbkeys.namedictstr: 'Fake numpy input attribute',
      'avg': 0,
      'stdev': stddev,
      'scaleratio': 1
    }]

  if applypars['img2img'] and args['infer_size'] is not None:
    serverpars['infer_size'] = applypars[dgbkeys.inpshapedictstr]

  if 'apply_dir' in applypars:
    serverpars['apply_dir'] = applypars['apply_dir']

  if args['fakeapply']:
    serverpars['fake_apply'] = True

  return serverpars

def getApplyTrace( dict ):
  platform = dict[dgbkeys.plfdictstr]
  img2img = dict['img2img']
  arr3d = dict['arr']
  shape = dict[dgbkeys.inpshapedictstr]
  if img2img:
    nrattribs = arr3d.shape[1]
    batchsize = dict['batchsize']
    batchidy = dict['idy']
    istart = batchidy*batchsize
    if batchsize == 1:
      ret = (arr3d[istart]).reshape((nrattribs,*shape))
    else:
      istop = (batchidy+1)*batchsize
      ret = (arr3d[istart:istop]).reshape((batchsize,nrattribs,*shape))
    return ret
  else:
    idxstart = dict['idx']
    idystart = dict['idy']
    idxstop = idxstart + (shape[0] if shape[0]>1 else 1)
    idystop = idystart + (shape[1] if shape[1]>1 else 1)
    ret = arr3d[:,idxstart:idxstop,idystart:idystop,:]
    if platform == dgbkeys.scikitplfnm:
      ret = np.reshape( ret, (len(ret),-1) )
    return ret

def create_request(action, value=None):
  if value == None:
    return dict(
      type="text/json",
      encoding="utf-8",
      content=dict(action=action),
    )
  elif action == 'parameters':
    return dict(
      type="text/json",
      encoding="utf-8",
      content=dict(
        action=action,
        value = value
      )
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

def wait_for_port(sockfam, addr, timeout=20.0):
  start = time.time()
  while time.time() - start < timeout:
    with socket.socket(sockfam, socket.SOCK_STREAM) as s:
      s.settimeout(0.2)
      if s.connect_ex(addr) == 0:
        return True
    time.sleep(0.05)
  return False

def req_connection(serverproc, sel, sockfam, addr, request):
  sock = socket.socket(sockfam, socket.SOCK_STREAM)
  sock.setblocking( False )
  sock.connect_ex(addr)

  message = applylib.Message(sel, sock, addr, request)
  events = selectors.EVENT_READ | selectors.EVENT_WRITE
  sel.register(sock, events, data=message)

  servererror = None
  ret = True
  while servererror is None:
    if serverproc.poll() is not None:
      std_msg( f'Apply server exited or crashed' )
      break

    events = sel.select(timeout=1)
    for key, mask in events:
      message = key.data
      try:
        message.process_events(mask)
      except Exception:
        std_msg(
            "main: error: exception for",
            f"{message._addr}:\n{traceback.format_exc()}",
        )
        message.close()
        ret = False
      finally:
        servererror = message._serverexception
        if mask & selectors.EVENT_READ:
          killreq = message._killreqhandled
          if servererror is not None:
            std_msg( f'server error: {servererror}' )
            break
          if killreq:
            break
#          res = message._response

    # Check for a socket being monitored to continue.
    if not sel.get_map():
      break

  return ret if servererror is None else False

def main() -> int:
  args = vars(parse_args())
  initLogging( args )

  pars = get_client_apply_pars( args )
  platform = pars[dgbkeys.plfdictstr]
  img2img = pars['img2img']
  inpdatashp = pars['inpdata_size']
  shape = pars[dgbkeys.inpshapedictstr]
  batchsize = pars['batchsize']
  rng = np.random.default_rng()
  stddev = 2500
  inpdata = stddev * rng.standard_normal(size=inpdatashp,dtype=np.single)

  host, port = args['addr'], args['port']
  if port == None:
    port = rng.integers(60000, high=65432, dtype=np.uint16)
  if host == 'LOCAL':
    sockfam = socket.AF_UNIX
    addr = str(port)
  else:
    sockfam = socket.AF_INET
    addr = (host, port)

  servercmd = get_server_cmd( args, port )
  serverproc = oscommand.execCommand( servercmd, background=True )
  if not wait_for_port(sockfam, addr):
    std_msg( 'Server never opened its port' )
    serverproc.terminate()
    serverproc.wait()
    return 1

  start = None
  sel = selectors.DefaultSelector()
  ex_code = 0
  nrtrcs = None
  try:
    if req_connection(serverproc, sel, sockfam, addr, create_request('status')):
      if req_connection(serverproc, sel, sockfam, addr, create_request('parameters', \
                        value=get_server_pars(args,pars,stddev))):
        start = time.time()
        applydict = {
          'arr': inpdata,
          dgbkeys.inpshapedictstr: shape,
          dgbkeys.plfdictstr: platform,
          'img2img': img2img
        }

        if img2img:
          nrimages = args['nr_images']
          nrtrcs = nrimages * batchsize
          applydict['batchsize'] = batchsize
          for idy in range(nrimages):
            applydict['idy'] = idy
            if not req_connection(serverproc, sel, sockfam, addr, create_request('data',applydict)):
              ex_code = 1
              break
        else:
          inlstep = args['inlstep']
          crlstep = args['crlstep']
          outpdatashp = pars['outpdata_size']
          nrinls_out = outpdatashp[0]
          nrtrcs_out = outpdatashp[1]
          inlrg = range(0,nrinls_out,inlstep)
          trcrg = range(0,nrtrcs_out,crlstep)
          nrtrcs = len(inlrg)*len(trcrg)
          for idx in inlrg:
            applydict['idx'] = idx
            for idy in trcrg:
              applydict['idy'] = idy
              if not req_connection(serverproc, sel, sockfam, addr, create_request('data',applydict)):
                ex_code = 1
                break

        if not req_connection(serverproc, sel, sockfam, addr, create_request('kill')):
          ex_code = 1
      else:
        ex_code = 1
        req_connection(serverproc, sel, sockfam, addr, create_request('kill'))
    else:
      ex_code = 1

  except KeyboardInterrupt:
    std_msg("caught keyboard interrupt, exiting")
    ex_code = 130
  except Exception:
    std_msg( f"Unexpected error: {traceback.format_exc()}" )
  finally:
    sel.close()
    if start is not None and nrtrcs is not None:
      duration = time.time()-start
      suffix = 'img' if img2img else 'tr'
      log_msg( f'Total time: {duration:.3f}s.; {nrtrcs/duration:.3f} {suffix}/s' )
    time.sleep( 1 )
    if serverproc and serverproc.poll() is None:
      oscommand.kill( serverproc )
  return ex_code

if __name__ == '__main__':
  sys.exit(main())
