#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : Wayne Mogg
# DATE     : January 2020
#
# Support methods for embedded Bokeh Server
#
#
import argparse
import os
import psutil
import sys
from bokeh.util.logconfig import basicConfig
from bokeh.application import Application
from bokeh.application.handlers import DirectoryHandler, ScriptHandler
from bokeh.io import curdoc
from bokeh.server.server import Server
import tornado.ioloop
import odpy.common as odcommon
import dgbpy.servicemgr as dgbservmgr

def get_request_id():
  reqargs = curdoc().session_context.request.arguments
  try:
    bokehid = int(reqargs.get('bokehid')[0])
  except:
    bokehid = -1
  return bokehid

def DefineBokehArguments(parser):
  netgrp = parser.add_argument_group( 'Network' )
  netgrp.add_argument( '--bsmserver',
            dest='bsmserver', action='store',
            type=str, default='',
            help='Bokeh service manager connection details' )
  netgrp.add_argument( '--ppid',
            dest='ppid', action='store',
            type=int, default=-1,
            help='PID of the parent process' )
  bokehgrp = parser.add_argument_group( 'Bokeh' )
  bokehgrp.add_argument( '--log-file',
            dest='bokehlogfnm', action='store',
            type=str, default='',
            help='Bokeh log-file name')
  bokehgrp.add_argument( '--address',
            dest='address', action='store',
            type=str, default='localhost',
            help='Bokeh server address')
  bokehgrp.add_argument( '--port',
            dest='port', action='store',
            type=int, default=5006,
            help='Bokeh server port')
  bokehgrp.add_argument( '--apps',
            dest='apps', action='store', default='bokeh_apps/crossplot',
            type=str,
            help='A list of apps to serve')
  bokehgrp.add_argument( '--show',
            dest='show', action='store_true', default=False,
            help='Show the app in a browser')
  return parser

def _getDocUrl(server):
  address_string = 'localhost'
  if server.address is not None and server.address != '':
    address_string = server.address
  url = "http://%s:%d%s" % (address_string, server.port, server.prefix)
  return url

class ProcMonitor:
  def __init__(self, ppid):
    self._parentproc = None
    self._bokehlog = None
    self._bokehserver = None
    if ppid > 0:
      try:
        self._parentproc = psutil.Process(ppid)
      except (ProcessLookupError, psutil.NoSuchProcess):
        raise ProcessLookupError

  def start(self, server, logfnm):
      self._bokehserver = server
      self._bokehlog = logfnm
      self.PCB = tornado.ioloop.PeriodicCallback(self._procChkCB, 1000)
      self.PCB.start()

  def _procChkCB(self):
    if self._parentproc and not self._parentproc.is_running():
      odcommon.log_msg('Found dead parent, exiting')
      self.PCB.stop()
      basicConfig(filename=None,force=True)
      if self._bokehserver != None:
        self._bokehserver.stop( True )
        if self._bokehlog != None and os.path.exists(self._bokehlog):
          os.remove( self._bokehlog )
      sys.exit()

parent_proc_checker = None
def StartBokehServer(applications, args, attempts=20):
  try:
    parent_proc_checker = ProcMonitor(args['ppid'])
  except ProcessLookupError:
    sys.exit()

  bokehlog = args['bokehlogfnm']
  basicConfig(filename=bokehlog)
  address = args['address']
  port = args['port']
  application = list(applications.keys())[0]
  while attempts:
    attempts -= 1
    try:
      authstr = f"{address}:{port}"
      server = Server(applications,address=address,
                      port=port,
                      allow_websocket_origin=[authstr,authstr.lower()])
      server.start()

      msg = dgbservmgr.Message()
      for app in list(applications.keys()):
        msg.sendObjectToAddress(args['bsmserver'], 'bokeh_started',
                                        { 'bokehapp': app,
                                          'bokehurl': _getDocUrl(server),
                                          'bokehpid': os.getpid()
                                        }
                                )
        if args['show']:
          server.show( app )

      try:
          parent_proc_checker.start( server, bokehlog )
          server.io_loop.start()
      except RuntimeError:
          pass
      return
    except OSError as ex:
      if "Address already in use" in str(ex):
        port += 1
      else:
        raise ex

  raise Exception("Failed to find available port")

def main():
  parser = argparse.ArgumentParser()
  parser = DefineBokehArguments(parser)
  bokehargs = vars(parser.parse_args())

  apps = bokehargs['apps'].split(',')
  applications = {}
  dirnm = os.path.dirname(__file__)
  for app in apps:
    file = os.path.join(dirnm,app)
    handler = None
    if os.path.isdir(file):
      handler = DirectoryHandler(filename=file)
    else:
      handler = ScriptHandler(filename=file)

    application = Application(metadata=bokehargs)
    application.add(handler)

    route = handler.url_path()
    applications[route] = application

  StartBokehServer(applications, bokehargs)

if __name__ == "__main__":
    main()
