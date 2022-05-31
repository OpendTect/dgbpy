# -*- coding: utf-8 -*-
"""
============================================================
Log plotting GUI
============================================================

 Author:    Wayne Mogg
 Copyright: dGB Beheer BV
 Date:      June 2020


"""
import argparse

import numpy as np
from bokeh.io import curdoc
import bokeh.layouts as bl
import bokeh.models as bm
import dgbpy.uibokeh_well as odb
import odpy.common as odcommon

undef = 1e30
survargs= odcommon.getODArgs()
wellnm = None
logs = []

def logplot_app(doc):
  well = odb.Well(wellnm, args=survargs)
  ltmgr = odb.LogTrackMgr(well, deflogs=logs, trackwidth=400, withui=True)
  doc.add_root(ltmgr.tracklayout)
  doc.title = 'Plot well'

logplot_app(curdoc())
