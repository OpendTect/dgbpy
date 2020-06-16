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

import sys
import argparse
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
datagrp.add_argument( '--welllogs',
            dest='welllogs', nargs=1,
            help='List of well logs to show' )
loggrp = parser.add_argument_group( 'Logging' )
loggrp.add_argument( '--proclog',
            dest='logfile', metavar='file', nargs='?',
            type=argparse.FileType('w'), default=sys.stdout,
            help='Progress report output' )
loggrp.add_argument( '--syslog',
            dest='sysout', metavar='stdout', nargs='?',
            type=argparse.FileType('w'), default=sys.stdout,
            help='Standard output' )

parser = DefineBokehArguments(parser)

args = vars(parser.parse_args())

import odpy.common as odcommon
odcommon.initLogging( args )

from odpy import wellman
import bokeh_logplot as blp

reload = False

wellid = args['wellid'][0]
blp.wellnm = wellman.getName( wellid, reload, args )
#blp.logs = args['welllogs']
blp.logs=['Sonic','Gamma Ray','Density']

StartBokehServer({'/': blp.logplot_app}, args)
