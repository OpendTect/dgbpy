# -*- coding: utf-8 -*-
"""
============================================================
Log plotting GUI
============================================================

 Author:    Paul de Groot <paul.degroot@dgbes.com>
 Copyright: dGB Beheer BV
 Date:      March 2020
 License:   GPL v3


@author: paul
"""

from os import path
import argparse
import odpy.wellman as wellman
import bokeh_logplot as blp
from dgbpy.bokehserver import StartBokehServer, DefineBokehArguments

parser = argparse.ArgumentParser(
            description='Dashboard for well log plotting')
parser.add_argument( '-v', '--version',
            action='version', version='%(prog)s 1.0' )
datagrp = parser.add_argument_group( 'Data' )
datagrp.add_argument( '--dataroot',
            dest='dtectdata', metavar='DIR', nargs=1,
            help='Survey Data Root' )
datagrp.add_argument( '--survey',
            dest='survey', nargs=1,
            help='Survey name' )
datagrp.add_argument( '--well',
            dest='wellid', nargs=1,
            help='Well ID' )

parser = DefineBokehArguments(parser)

args = vars(parser.parse_args())
reload = False

wellid = args['wellid'][0]
blp.wellnm = wellman.getName( wellid, reload, args )

(blp.headers,blp.data,blp.mindepth,blp.maxdepth) = blp.readLogs( blp.wellnm, blp.undef, reload, args )
depthlogstr = blp.headers[0]
blp.loglist = blp.updateLogList(depthlogstr, blp.headers, option = 1)
blp.logonlylist = blp.updateLogList(depthlogstr, blp.headers, option = 0)

StartBokehServer({'/': blp.logplot_app}, args)
