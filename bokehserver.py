#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : Wayne Mogg
# DATE     : January 2020
#
# Support methods for embedded Bokeh Server
#
# 
import os
from bokeh.util.logconfig import basicConfig
from bokeh.server.server import Server
import dgbpy.servicemgr as dgbservmgr

def DefineBokehArguments(parser):
  netgrp = parser.add_argument_group( 'Network' )
  netgrp.add_argument( '--bsmserver',
            dest='bsmserver', action='store',
            type=str, default='',
            help='Bokeh service manager connection details')
  netgrp.add_argument( '--bokehid',
            dest='bokehid', action='store',
            type=int, default=-1,
            help='Bokeh server id')
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
            type=str, default='',
            help='Bokeh server address')
  bokehgrp.add_argument( '--port',
            dest='port', action='store',
            type=int, default=-1,
            help='Bokeh server port')
  return parser

def _getDocUrl(server, app_path):
  address_string = 'localhost'
  if server.address is not None and server.address != '':
    address_string = server.address
  url = "http://%s:%d%s%s" % (address_string, server.port, server.prefix, app_path)
  return url

def StartBokehServer(application, args, attempts=20):
  basicConfig(filename=args['bokehlogfnm'])
  address = args['address']
  port = args['port']
  while attempts:
    attempts -= 1
    try:
      authstr = f"{address}:{port}"
      server = Server(application,address=address,
                      port=port,
                      allow_websocket_origin=[authstr,authstr.lower()])
      server.start()
      
      msg = dgbservmgr.Message()
      msg.sendObjectToAddress(args['bsmserver'],
                      'bokeh_started', {'bokehid': args['bokehid'],
                                        'bokehurl': _getDocUrl(server,"/"),
                                        'bokehpid': os.getpid()})

      server.io_loop.start()
      return
    except OSError as ex:
      if "Address already in use" in str(ex):
        port += 1
      else:
        raise ex
      
  raise Exception("Failed to find available port")
    

