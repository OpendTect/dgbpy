from bokeh.io import curdoc
from well_tools import MultiWellCrossPlot

xplot = MultiWellCrossPlot()
curdoc().add_root(xplot.get_controls())
curdoc().title = "Multi-well Crossplot"
