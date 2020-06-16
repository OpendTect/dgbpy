"""
============================================================
Log plotting using Bokeh
============================================================

 Author:    Wayne Mogg
 Copyright: dGB Beheer BV
 Date:      June 2020
 

"""
import odpy.wellman as odwm
import bokeh.plotting as bp
import bokeh.models as bm
import bokeh.layouts as bl
import math
import numpy as np
from odpy.ranges import niceRange

class Well:
    def __init__(self,wellname):
        self.wellname = wellname
        self.track = None
        self.markers = None
        self.logcache = {}

    def getLogNames(self):
        return odwm.getLogNames(self.wellname)
    
    def getLog(self, lognm):
        if lognm not in self.logcache:
            (depths,logvals) = odwm.getLog(self.wellname, lognm)
            self.logcache[lognm] = bm.ColumnDataSource({'depths': depths, lognm: logvals})
        return self.logcache[lognm]

    def getTrack(self):
        if not self.track:
            self.track = odwm.getTrack(self.wellname)
        return self.track
    
    def getMarkers(self):
        if not self.markers:
            (mrkrs, depths, colors) = odwm.getMarkers(self.wellname)
            self.markers = bm.ColumnDataSource({'name': mrkrs, 'depth': depths, 'color': colors})
        return self.markers
    
    def depthRange(self):
        return self.getTrack()[0]
        
class LogTrack:
    def __init__(self, well, width, color='#f0f0f0'):
        self.well = well
        self.width = width
        self.back_color = color
#        self.shading =np.empty((height,width,4),dtype=np.uint8)
        self._inittrack()
        self.logcache = {}
        
    def _inittrack(self):
        depths = self.well.depthRange()
        tooltips = [
            ("Depth", " $y"),
            ("Log","$x"),
        ]
        self.track = bp.figure(title=self.well.wellname,
                             plot_width=self.width,
                             sizing_mode='stretch_height',
                             background_fill_color=self.back_color,
                             tools='ypan,ywheel_zoom,reset,hover',
                             y_axis_label='Depth (mMD)',
                             tooltips=tooltips)
        self.track.y_range = bm.Range1d(depths[-1], depths[0],bounds='auto')
        self.track.xaxis.visible = False
        
    def addLog(self, lognm, color):
        log = self.well.getLog(lognm)
        limits = niceRange(min(log.data[lognm]), max(log.data[lognm]))
        axis_style = {'axis_line_color': color,
                      'axis_line_width': 2,
                      'axis_line_dash': 'solid',
                      'major_tick_line_color': color,
                      'major_tick_line_width': 2,
                      'major_tick_in': 0,
                      'axis_label_standoff': -10
                     }
        log_style = {'line_color': color,
                     'line_width': 2,
                     'line_dash': 'solid'
                    }
        self.track.extra_x_ranges[lognm] = bm.Range1d(limits[0], limits[-1])
        self.track.add_layout(bm.LinearAxis(x_range_name=lognm,axis_label=lognm,ticker=limits,**axis_style), 'above')
        self.track.line(x=lognm, y='depths',x_range_name=lognm,source=log, **log_style)
        
    def addLogs(self, logs, colors):
        lognms = self.well.getLogNames()
        showlogs = set.intersection(set(logs), set(lognms))
        for lognm in showlogs:
            color = colors[logs.index(lognm)]
            self.addLog(lognm, color)
            
    def addMarker(self, name, depth, color):
        (lognm, xr) = next(iter(self.track.extra_x_ranges.items()),(None,None))
        if not lognm:
            return
        marker_style = {'line_color': color,
                        'line_width': 2,
                        'line_dash': 'solid'
                       }
        self.track.add_layout(bm.Arrow(end=None,x_start=xr.start,y_start=depth,x_end=xr.end,y_end=depth,x_range_name=lognm,**marker_style))
        self.track.add_layout(bm.Label(x=xr.start,y=depth,x_range_name=lognm,text=name,text_color=color,render_mode='canvas'))
        
    def addMarkers(self):
        markers = self.well.getMarkers().data
        for (name,depth,color) in zip( markers['name'],markers['depth'],markers['color']):
            self.addMarker(name,depth,color)
        
    def clearShading(self):
        self.shading = self.back_color
        
    def addLeftShade(self,log,color):
        pass
         
