"""
============================================================
Log plotting using Bokeh
============================================================

 Author:    Wayne Mogg
 Copyright: dGB Beheer BV
 Date:      June 2020
 

"""
import odpy.wellman as odwm
from bokeh.events import ButtonClick, Event
import bokeh.plotting as bp
import bokeh.models as bm
import bokeh.layouts as bl
import bokeh.core.enums as bce
from bokeh.core.properties import DashPattern
import math
import numpy as np
import random
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

class LogTrackMgr:
    def __init__(self, well, deflogs=None, trackwidth=400, withui=False):
        self.well = well
        self.width = trackwidth
        self.withui = withui
        self.tracks = {}
        self.tracklayout = None
        self.trackid = 0
        lognames = self.well.getLogNames()
        if not deflogs:
            self.deflogs = [lognames[0]]
        else:
            self.deflogs = list(set.intersection(set(deflogs), set(lognames)))
        self._addtrack()
        

    def _addtrack(self, copytrack=None):
        
        if self.withui:
            addbutton = bm.Button(label='+', button_type='success', width_policy='min')
            addbutton.on_event(ButtonClick, self._addbuttonCB)
            rmvbutton = bm.Button(label='-', button_type='warning', width_policy='min')
            rmvbutton.on_event(ButtonClick, self._rmvbuttonCB)
            buttons = bl.row(addbutton,rmvbutton, align='center')
            newtrack = LogTrack(self.well, self.width, withui=self.withui)
            if copytrack:
                newtrack.log_props = copytrack.log_props.copy()
                logs = copytrack.log_select.value
                newtrack.addLogs(logs)
            else:
                newtrack.addLogs(self.deflogs)
            trackbox = bl.column([newtrack.display(), buttons], name=str(self.trackid), tags=[addbutton.id, rmvbutton.id])
            if self.tracklayout:
                self.tracklayout.children.append(trackbox)
            else:
                self.tracklayout = bl.row([trackbox], sizing_mode='stretch_height')
                rmvbutton.visible = False
            self.tracks[str(self.trackid)] = newtrack
            self.trackid += 1
        else:
            self.tracklayout = LogTrack(self.well, self.width, withui=False)

    def _removetrack(self):
        pass

    def _addbuttonCB(self, event):
        evid = event._model_id
        for trackbox in self.tracklayout.children:
            if evid in trackbox.tags:
                self._addtrack(copytrack=self.tracks[trackbox.name])
                break
        if len(self.tracks)>1:
            trackbox = self.tracklayout.children[0]
            rmvbutton = trackbox.children[-1].children[-1]
            rmvbutton.visible = True
        
    def _rmvbuttonCB(self, event ):
        evid = event._model_id
        for idx in range(0, len(self.tracklayout.children)):
            trackbox = self.tracklayout.children[idx]
            if evid in trackbox.tags:
                self.tracks.pop(trackbox.name)
                self.tracklayout.children.pop(idx)
                break
        if len(self.tracks)==1:
            trackbox = self.tracklayout.children[0]
            rmvbutton = trackbox.children[-1].children[-1]
            rmvbutton.visible = False
 
class LogTrack:
    def __init__(self, well, width, color='#f0f0f0', withui=False):
        self.well = well
        self.width = width
        self.back_color = color
        self.withui = withui
        self.tracklayout = None
        self.lc = {}
        self.log_props = {}
        self._initlogprops()
#        self.shading =np.empty((height,width,4),dtype=np.uint8)
        self._inittrack()
        self._initui()
        
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
        self.track.title.text_font_size = '14pt'
        self.track.y_range = bm.Range1d(depths[-1], depths[0],bounds='auto')
        self.track.xaxis.visible = False

    def _initlogprops(self):
        lognames = self.well.getLogNames()
        for log in lognames:
            self.log_props[log] = {'left': None,
                                   'right': None,
                                   'lwidth': 2,
                                   'lcolor': "#%06x" % random.randint(0,0xFFFFFF),
                                   'ldash': 'solid'
                                   }


    def _log_control(self):
        lognames = self.well.getLogNames()
        self.lc = { 'log': bm.Select(title='Log Display Styling:', options=lognames),
                  'left': bm.Spinner(title='Left Limit:'),
                  'right': bm.Spinner(title='Right Limit:'),
                  'width': bm.Spinner(title='Line Width:',low=0,high=5, step=1),
                  'color': bm.ColorPicker(title='Line Color:'),
                  'line': bm.Select(title='Line Style:', options=list(bce.DashPattern))
                }
        self.lc['log'].on_change('value', self._update_log_control)
        for (key, item) in self.lc.items():
          if key=='log':
            continue
          if key=='color':
            item.on_change('color', self._update_log_properties)
          else:
            item.on_change('value', self._update_log_properties)
        self.log_select.js_link('value',self.lc['log'], 'options')
        return bl.layout(self.lc['log'],self.lc['left'],self.lc['right'],self.lc['width'],self.lc['color'],self.lc['line'], align='center')
      
    def _initui(self):
        lognames = self.well.getLogNames()
        self.log_select = bm.MultiSelect(title="Logs:", options=lognames, align='center')
        self.log_select.on_change('value', self._update_track)
        self.marker_select = bm.MultiSelect(title="Markers:", options=self.well.getMarkers().data['name'], align='center')
        self.marker_select.on_change('value', self._update_track)
        self.ui = bl.column(self.log_select, self.marker_select, self._log_control(), align=('center','start'))
      
    def _get_log_settings(self):
        ls = {}
        
    def _update_log_control(self, attr, new, old):
        lognm = self.lc['log'].value
        start = self.track.extra_x_ranges[lognm].start
        end = self.track.extra_x_ranges[lognm].end
        line = self.track.select(name=lognm).glyph
        width = line.line_width
        color = line.line_color
        linedash = line.line_dash
        self.lc['left'].update(value=start)
        self.lc['right'].update(value=end)
        self.lc['width'].update(value=width)
        self.lc['color'].update(color=color)
        self.lc['line'].update(value='solid')
        for (key, value) in DashPattern._dash_patterns.items():
         if value==linedash:
            self.lc['line'].update(value=key)
            break

    def _update_log_properties(self, attr, new, old):
        lognm = self.lc['log'].value
        start = self.lc['left'].value
        end = self.lc['right'].value
        color = self.lc['color'].color
        width = int(self.lc['width'].value)
        dash = self.lc['line'].value
        self.log_props[lognm] = {'left': start,
                                 'right': end,
                                 'lwidth': width,
                                 'lcolor': color,
                                 'ldash': dash
                                }
        self.track.extra_x_ranges[lognm].update(start=start, end=end)                    
        line = self.track.select(name=lognm).glyph
        line.update(line_width=width, line_color=color, line_dash=dash)
        axis = self.track.select(axis_label=lognm)
        axis.update(axis_line_color=color, major_tick_line_color=color,
                    axis_line_width=width,major_tick_line_width=width,
                    axis_line_dash=dash, ticker=(start, end))
        
    def _update_track(self, attr, old, new):
        if self.withui:
            self._inittrack()
            self.addLogs(self.log_select.value)
            self.addMarkers(self.marker_select.value)
            if self.tracklayout:
                self.tracklayout.children[0] = self.track            
        
    def display(self):
        if self.withui:
          self.tracklayout = bl.column(self.track)
          tracktab = bm.Panel(child=self.tracklayout, title="Plot")
          uitab = bm.Panel(child=self.ui, title="Parameters")
          return bm.Tabs(tabs=[tracktab, uitab])
        else:
          return self.track
      
    def addLog(self, lognm):
        log = self.well.getLog(lognm)
        limits = None
        left = self.log_props[lognm]['left']
        right = self.log_props[lognm]['right']
        if left and right:
            limits = (left, right)
        else:
            limits = niceRange(min(log.data[lognm]), max(log.data[lognm]))
            self.log_props[lognm]['left'] = limits[0]
            self.log_props[lognm]['right'] = limits[-1]
            
        lwidth = self.log_props[lognm]['lwidth']
        lcolor = self.log_props[lognm]['lcolor']
        ldash = self.log_props[lognm]['ldash']

        axis_style = {'axis_line_color': lcolor,
                      'axis_line_width': lwidth,
                      'axis_line_dash': ldash,
                      'major_tick_line_color': lcolor,
                      'major_tick_line_width': lwidth,
                      'major_tick_in': 0,
                      'axis_label_standoff': -10
                     }
        log_style = {'line_color': lcolor,
                     'line_width': lwidth,
                     'line_dash': ldash
                    }
        self.track.extra_x_ranges[lognm] = bm.Range1d(limits[0], limits[-1])
        self.track.add_layout(bm.LinearAxis(x_range_name=lognm,axis_label=lognm,ticker=limits,**axis_style), 'above')
        self.track.line(x=lognm, y='depths',x_range_name=lognm,source=log, name=lognm,  **log_style)
        
    def addLogs(self, logs):
        lognms = self.well.getLogNames()
        showlogs = list(set.intersection(set(logs), set(lognms)))
        for lognm in showlogs:
            self.addLog(lognm)
        if self.withui:
            self.log_select.update(value=showlogs)
            self.lc['log'].update(options=showlogs, value=showlogs[0])
            
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
        
    def addMarkers(self, markers=None):
        mrksource = self.well.getMarkers().data
        showmarkers = []
        for (name,depth,color) in zip( mrksource['name'],mrksource['depth'],mrksource['color']):
            if not markers or name in markers:      
                self.addMarker(name,depth,color)
                showmarkers += [name]
        if self.withui:
          self.marker_select.update(value=showmarkers)
        
    def clearShading(self):
        self.shading = self.back_color
        
    def addLeftShade(self,log,color):
        pass
         
    