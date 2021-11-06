# -*- coding: utf-8 -*-
"""
============================================================
Cubelets Plotting GUI
============================================================

 Author:    Paul de Groot, Arnaud Huck
 Copyright: dGB Beheer BV
 Date:      May 2020
 License:   GPL v3


@author: paul
"""
import sys
import argparse
from dgbpy.bokehserver import StartBokehServer, DefineBokehArguments

parser = argparse.ArgumentParser(
            description='Dashboard for ML examples plotting')
parser.add_argument( '-v', '--version',
            action='version', version='%(prog)s 2.0' )
parser.add_argument( 'h5file',
            type=argparse.FileType('r'),
            help='HDF5 file containing the training data' )
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

import bokeh_plot_examples as bpe

bpe.exfilenm = args['h5file'].name
StartBokehServer({'/': bpe.exampleplot_app}, args)
