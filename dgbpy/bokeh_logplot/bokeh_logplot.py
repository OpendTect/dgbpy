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
import dgbpy.uibokeh_well as odb
import odpy.common as odcommon

undef = 1e30
survargs= odcommon.getODArgs()
wellnm = 'None'
logs = []

def logplot_app(doc):
  well = odb.Well(wellnm, args=survargs)
  ltmgr = odb.LogTrackMgr(well, deflogs=logs, trackwidth=400, withui=True)
  doc.add_root(ltmgr.tracklayout)
  doc.title = 'Plot well'


def main():
  global survargs, wellnm, logs

  survargs = {'dtectdata': ['/mnt/Data/seismic/ODData'], 'survey': ['F3_Demo_2020']}
  wellnm = 'F03-2'
  logs = ['Sonic','Gamma Ray','P-Impedance_rel']

  server = Server({'/' : logplot_app})
  server.start()
  server.io_loop.add_callback(server.show, "/")
  server.io_loop.start()

if __name__ == "__main__":
    main()
