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
from bokeh.server.server import Server
import bokeh.layouts as bl
import bokeh.models as bm
import odbokeh as odb

undef = 1e30
wellnm = 'None'
logs = []

def logplot_app(doc):
  well = odb.Well(wellnm)
  ltmgr = odb.LogTrackMgr(well, deflogs=logs, trackwidth=400, withui=True)
  doc.add_root(ltmgr.tracklayout)
  doc.title = 'Plot well'


def main():
  global wellnm, logs
  
  wellnm = 'F02-1'
  logs = ['Sonic','Gamma Ray','P-Impedance_rel']
  
  server = Server({'/' : logplot_app})
  server.start()
  server.io_loop.add_callback(server.show, "/")
  server.io_loop.start()

if __name__ == "__main__":
    main()
