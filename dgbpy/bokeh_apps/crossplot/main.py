from bokeh.io import curdoc
from well_tools import MultiWellCrossPlot
import logging
import odpy.common as odcommon
from functools import partial

odcommon.proclog_logger = logging.getLogger('bokeh.bokeh_machine_learning.main')
odcommon.proclog_logger.setLevel( 'DEBUG' )

def crossplotapp(doc):
  xplot = MultiWellCrossPlot()
  def bokehParChgCB(paramobj):
    nonlocal xplot
    wells = paramobj.get("Well") 
    if wells:
      doc.add_next_tick_callback(partial(xplot.select_wells,wellnms=wells))

  doc.add_root(xplot.get_controls())
  doc.title = "Multi-well Crossplot"
  args = doc.session_context.server_context.application_context.application.metadata
  if args:
    from dgbpy.bokehserver import get_request_id
    from dgbpy.servicemgr import ServiceMgr
    with ServiceMgr(args['bsmserver'],args['ppid'],args['port'],get_request_id()) as this_service:
      this_service.addAction('BokehParChg', bokehParChgCB )
  else:
    from sys import argv
    import json
    if  len(argv)>1:
      data = json.loads(argv[1])
      bokehParChgCB(data)

crossplotapp(curdoc())

