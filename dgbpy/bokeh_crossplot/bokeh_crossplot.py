# -*- coding: utf-8 -*-
"""
============================================================
Log crossplotting GUI
============================================================

 Author:    Wayne Mogg
 Copyright: dGB Beheer BV
 Date:      June 2020


"""
import argparse
from os.path import dirname,join
from functools import partial

import pandas as pd
import numpy as np
from sklearn import linear_model
from bokeh.server.server import Server
import bokeh.layouts as bl
import bokeh.models as bm
import bokeh.plotting as bp
import bokeh.core.enums as bce
from bokeh.transform import linear_cmap
from bokeh.core.properties import DashPattern, value
import bokeh.palettes as bpal
from bokeh.events import Reset, MouseWheel, PanEnd

import dgbpy.uibokeh_well as odb
import odpy.common as odcommon

undef = 1e30
survargs= odcommon.getODArgs()
wellnm = 'None'
welllogs = '0'

class LogRangeWidget:
    def __init__(self, well, width=250, title='Extract over: '):
        self.well = well
        markernms = self.well.getMarkers().data['name']
        xmarkernms = ['<Start of Data>'] + markernms + ['<End of Data>']
        field_title = bm.Div(text=title)
        hwidth = int(width/2-10)
        self.fields = {
            'range': bm.RadioGroup(labels=['Depth Range','Markers'], inline=True,
                                   active=1, width_policy='min'),
            'zupper': bm.Spinner(title='From depth', width=hwidth),
            'mrkupper': bm.Select(title='Relative to Marker', value=xmarkernms[0],
                                  options=xmarkernms, width=hwidth),
            'zlower': bm.Spinner(title ='To depth', width=hwidth),
            'mrklower': bm.Select(title='Relative to Marker', value=xmarkernms[-1],
                                  options=xmarkernms, width=hwidth),
            }
        for (key,field) in self.fields.items():
            if key=='range':
                field.on_change('active', self.rangeTypeChg)
            else:
                field.on_change('value', self.depthChg)
        self.layout = bl.column(bl.row(field_title, self.fields['range']),
                                bl.row(self.fields['zupper'], self.fields['mrkupper']),
                                bl.row(self.fields['zlower'], self.fields['mrklower']),
                                width=width
                                )
    def show(self, yn):
        for field in self.fields.items():
            field.visible = yn

    def reset(self):
        if self.fields['range'].active==0:
            depths = self.well.depthRange()
            self.fields['zupper'].update(value=depths[0])
            self.fields['zlower'].update(value=depths[-1])
        else:
            self.fields['zupper'].update(value=0)
            self.fields['zlower'].update(value=0)
            self.fields['mrkupper'].update(value=0)
            self.fields['mrklower'].update(value=-1)
        self.depthChg(None, None, None)

    def rangeTypeChg(self, attr, old, new):
        if new == 0:
            self.fields['mrkupper'].visible = self.fields['mrklower'].visible = False
            depths = self.well.depthRange()
            self.fields['zupper'].update(value=depths[0])
            self.fields['zlower'].update(value=depths[-1])
        else:
            self.fields['mrkupper'].visible = self.fields['mrklower'].visible = True
            self.fields['zupper'].update(value=0)
            self.fields['zlower'].update(value=0)
        self.depthChg(None, None, None)

    def depthChg(self, attr, old, new):
        depths = list(self.well.depthRange())
        if self.fields['range'].active==0:
            depths = [self.fields['zupper'].value, self.fields['zlower'].value]
        else:
            mrkupper = self.fields['mrkupper'].value
            mrklower = self.fields['mrklower'].value
            markers = self.well.getMarkers().data
            if mrkupper!='<Start of Data>':
                depths[0] = markers['depth'][markers['name'].index(mrkupper)]+self.fields['zupper'].value
            if mrklower!='<End of Data>':
                depths[1] = markers['depth'][markers['name'].index(mrklower)]+self.fields['zlower'].value
        self.well.setDepthView(min(depths), max(depths))

class ColorMapWidget:
    def __init__(self, title='Color Map' ):
        self.cmaps = []
        for (key,val) in bpal.all_palettes.items():
            if 256 in list(val.keys()):
                self.cmaps += [key]
        self.cmaps.sort()
        self.fields = {'colormap': bm.Select(title=title, value='Viridis', options=self.cmaps)}
        self.layout = self.fields['colormap']

class CrossplotControls:
    def __init__(self, well):
        self.well = well
        self.logrange = None
        self.fields = {}
        self._initui()

    def _initui(self):
        width = 300
        lognms = self.well.getLogNames()
        xlognms = ['None'] + lognms
        self.logrange = LogRangeWidget(self.well, width=width)
        cmap = ColorMapWidget()
        self.fields = {
            'wellnm': bm.TextInput(title='Well', value=wellnm ),
            'xlog': bm.Select(title='X-Axis Log', value=lognms[1], options=lognms),
            'ylog': bm.Select(title='Y-Axis Log', value=lognms[2], options=lognms),
            'colorlog': bm.Select(title='Color Log', value=xlognms[0], options=xlognms),
            'sizelog': bm.Select(title='Size Log', value=xlognms[0], options=xlognms),
            'sizemap': bm.RangeSlider(start=0, end=40, value=(6, 25), step=1, title='Size Map'),
            'markers': bm.CheckboxGroup(labels=['Show markers'], inline=True, active=[]),
            'regression': bm.CheckboxGroup(labels=['Show regression fit'], inline=True, active=[]),
            'selected': bm.CheckboxGroup(labels=['For selected only'], inline=True, active=[])
            }
        self.fields.update(cmap.fields)
        self.fields.update(self.logrange.fields)
        self.fields['wellnm'].disabled = True
        self.layout = bl.column(self.fields['wellnm'],
                                self.fields['xlog'],
                                self.fields['ylog'],
                                self.fields['colorlog'],
                                self.fields['colormap'],
                                self.fields['sizelog'],
                                self.fields['sizemap'],
                                self.fields['markers'],
                                self.logrange.layout,
                                self.fields['regression'],
                                self.fields['selected'],
                                width=width,
                                width_policy='fixed'
                                )

class CrossplotLogTracks:
    def __init__(self, well, logs, width=300):
        self.well = well
        self.width = width
        self.track_props = {}
        self.log_props = {}
        self.layout = None
        self.tooltips = [("Depth", " $y"),
                         ("Log","$x"),
                        ]

        self.datatrack = None
        self.disptrack = None
        self._init_log_props()
        self._init_track_props()
        self._inittrack()
        self._init_logs(logs)
        self.marker_props = None
        self._init_markers()

    def _inittrack(self):
        depths = self.well.depthRange()
        self.datatrack = bp.figure(title='Crossplot Logs',
                             plot_width=self.track_props['plot_width'],
                             sizing_mode='stretch_height',
                             background_fill_color=self.track_props['background_fill_color'],
                             tools='ypan,ywheel_zoom,reset,hover', active_scroll='ywheel_zoom',
                             toolbar_location='right',
                             y_axis_label='Depth (mMD)',
                             tooltips=self.tooltips,
                             min_border_right=20,
                             reset_policy=bce.ResetPolicy.event_only)
        box_zoom = bm.BoxZoomTool(dimensions="height")
        box_select = bm.BoxSelectTool(dimensions="height")
        self.datatrack.add_tools(box_zoom)
        self.datatrack.add_tools(box_select)
        self.datatrack.title.text_font_size = '12pt'
        self.datatrack.y_range = bm.Range1d(depths[-1], depths[0],bounds='auto')
        self.datatrack.x_range = bm.Range1d(0, 10)
        self.datatrack.xaxis.visible = False
        self.datatrack.on_event(Reset, self.resetCB)
        self.datatrack.on_event(MouseWheel, self.resetCB)
        self.datatrack.on_event(PanEnd, self.resetCB)

        self.disptrack = bp.figure(title='Color/Size Logs',
                             plot_width=self.track_props['plot_width'],
                             sizing_mode='stretch_height',
                             background_fill_color=self.track_props['background_fill_color'],
                             tools='ypan,ywheel_zoom,reset,hover', active_scroll='ywheel_zoom',
                             y_axis_label=' ',
                             tooltips=self.tooltips,
                             min_border_right=20,
                             reset_policy=bce.ResetPolicy.event_only)
        self.disptrack.add_tools(box_zoom)
        self.disptrack.add_tools(box_select)
        self.disptrack.title.text_font_size = '12pt'
        self.disptrack.y_range = self.datatrack.y_range
        self.disptrack.x_range = bm.Range1d(0, 10)
        self.disptrack.xaxis.visible = False
        self.apply_track_props(self.datatrack)
        self.apply_track_props(self.disptrack)
        self.layout = bl.gridplot([[self.datatrack, self.disptrack]],
                                  toolbar_location='left', sizing_mode='stretch_height')

    def _init_logs(self, logs):
        self.addLog(logs[0], 'xlog')
        self.addLog(logs[1], 'ylog')
        self.addLog(logs[0], 'sizelog')
        self.addLog(logs[1], 'colorlog')

    def _init_markers(self):
        if not self.marker_props:
            self.marker_props = {'style': {'line_width': 2, 'line_dash': 'solid', 'line_color': 'red'}}
        markers = self.well.getMarkers().data
        mrkprops = {}
        mrkprops.update(self.marker_props['style'])
        for (name,depth,color) in zip( markers['name'],markers['depth'],markers['color']):
            mrkprops['line_color'] = color
            xr = self.datatrack.x_range
            self.marker_props[name] = {'Line': bm.Arrow(end=None,x_start=xr.start,y_start=depth,
                                                        x_end=xr.end,y_end=depth,**mrkprops,
                                                        visible=False),
                                       'Label': bm.Label(x=xr.start,y=depth,text=name,
                                                         text_color=color,render_mode='canvas',
                                                         text_font_size='8pt',visible=False)
                                      }
            self.datatrack.add_layout(self.marker_props[name]['Line'])
            self.datatrack.add_layout(self.marker_props[name]['Label'])

    def resetCB(self,ev):
        if isinstance(ev, Reset):
            depths = self.well.depthRange()
            self.datatrack.y_range.update(start=depths[-1], end=depths[0])
            self.well.logdata.selected.indices = []

        lognm = self.log_props['xlog']['log']
        if not lognm:
            return
        self.updateLog(lognm, 'xlog')
        lognm = self.log_props['ylog']['log']
        if not lognm:
            return
        self.updateLog(lognm, 'ylog')

    def addLog(self, lognm, logtype='xlog'):
        width = self.log_props[logtype]['style']['line_width']
        color = self.log_props[logtype]['style']['line_color']
        dash = self.log_props[logtype]['style']['line_dash']
        alpha =  self.log_props[logtype]['style']['line_alpha']
        nsalpha =  self.log_props[logtype]['style']['nonselection_line_alpha']
        axis_style = {'axis_line_color': color,
                      'axis_line_width': width,
                      'axis_line_dash': dash,
                      'axis_line_alpha': alpha,
                      'major_tick_line_color': color,
                      'major_tick_line_width': width,
                      'major_tick_line_alpha': alpha,
                      'major_tick_in': 0,
                      'axis_label_standoff': 0
                     }
        log_style = {'line_color': color,
                     'line_width': width,
                     'line_dash': dash,
                     'line_alpha': alpha
                    }
        limits = self.well.getLogLimits(lognm)
        track = self.datatrack
        if logtype == 'sizelog' or logtype == 'colorlog':
            track = self.disptrack
        track.extra_x_ranges[logtype] = bm.Range1d(limits[0], limits[-1])
        self.log_props[logtype]['axis'] = bm.LinearAxis(x_range_name=logtype,axis_label=lognm,
                                                        ticker=limits, name=logtype, **axis_style)
        track.add_layout(self.log_props[logtype]['axis'], 'above')
        self.log_props[logtype]['line'] = track.line(x=lognm, y='depth',x_range_name=logtype,
                                                     source=self.well.logdata, name=logtype, **log_style,
                                                     view=self.well.logdataview)
        log_style['line_alpha'] = nsalpha
        self.log_props[logtype]['points'] = track.circle(x=lognm, y='depth',x_range_name=logtype,
                                                         size=1, color=None,
                                                         source=self.well.logdata,
                                                         view=self.well.logdataview)
        self.log_props[logtype]['line'].visible = self.log_props[logtype]['visible']
        self.log_props[logtype]['points'].visible = self.log_props[logtype]['visible']
        self.log_props[logtype]['axis'].visible = self.log_props[logtype]['visible']
        self.log_props[logtype]['log'] = lognm

    def updateLog(self, lognm, logtype):
        if not self.log_props[logtype]['axis']:
            self.addLog(lognm, logtype)
            return

        limits = self.well.getLogLimits(lognm)
        track = self.datatrack
        if logtype == 'sizelog' or logtype == 'colorlog':
            track = self.disptrack

        track.extra_x_ranges[logtype].update(start=limits[0], end=limits[-1])
        self.log_props[logtype]['axis'].update(axis_label=lognm, ticker=limits)
        self.log_props[logtype]['line'].glyph.update(x=lognm)
        self.log_props[logtype]['line'].visible = self.log_props[logtype]['visible']
        self.log_props[logtype]['axis'].visible = self.log_props[logtype]['visible']
        self.log_props[logtype]['log'] = lognm


    def showLog(self, logtype, yn):
        self.log_props[logtype]['visible'] = yn
        self.log_props[logtype]['line'].visible = self.log_props[logtype]['visible']
        self.log_props[logtype]['points'].visible = self.log_props[logtype]['visible']
        self.log_props[logtype]['axis'].visible = self.log_props[logtype]['visible']

    def show_markers(self, show):
        for (name,item) in self.marker_props.items():
            if name=='style':
                continue
            item['Line'].visible = show
            item['Label'].visible = show

    def _init_log_props(self):
        self.log_props['xlog'] = {'axis': None,
                                  'line': None,
                                  'log': None,
                                  'points': None,
                                  'style': {'line_width': 2,
                                            'line_color': 'darkorange',
                                            'line_dash': 'solid',
                                            'line_alpha': 1,
                                            'nonselection_line_alpha':0.3
                                            },
                                  'visible': True
                                 }
        self.log_props['ylog'] = {'axis': None,
                                  'line': None,
                                  'log': None,
                                  'points': None,
                                  'style': {'line_width': 2,
                                            'line_color': 'royalblue',
                                            'line_dash': 'solid',
                                            'line_alpha': 1,
                                            'nonselection_line_alpha':0.3
                                            },
                                  'visible': True
                                 }
        self.log_props['sizelog'] = {'axis': None,
                                     'line': None,
                                     'log': None,
                                     'points': None,
                                     'style': {'line_width': 2,
                                               'line_color': 'indigo',
                                               'line_dash': 'solid',
                                               'line_alpha': 1,
                                               'nonselection_line_alpha':0.3
                                              },
                                     'visible': False
                                    }
        self.log_props['colorlog'] = {'axis': None,
                                      'line': None,
                                      'log': None,
                                      'points': None,
                                      'style': {'line_width': 2,
                                                'line_color': 'forestgreen',
                                                'line_dash': 'solid',
                                                'line_alpha': 1,
                                                'nonselection_line_alpha':0.3
                                               },
                                      'visible': False
                                     }


    def _init_track_props(self):
        tmp = bp.figure()
        self.track_props = {'plot_width': self.width,
                            'background_fill_color': tmp.background_fill_color,
                            'log_major_visible': True,
                            'log_major_num': 2,
                            'log_major_width': 2,
                            'log_major_color': tmp.xgrid.grid_line_color,
                            'log_minor_visible': True,
                            'log_minor_width': 1,
                            'log_minor_color': tmp.xgrid.grid_line_color,
                            'log_minor_num': 5,
                            'z_major_visible': tmp.ygrid.visible,
                            'z_major_num': tmp.yaxis[0].ticker.desired_num_ticks-1,
                            'z_major_width': 2,
                            'z_major_color': tmp.ygrid.grid_line_color,
                            'z_minor_visible': tmp.ygrid.minor_grid_line_color is not None,
                            'z_minor_width': tmp.ygrid.minor_grid_line_width,
                            'z_minor_color': tmp.ygrid.grid_line_color,
                            'z_minor_num': tmp.yaxis[0].ticker.num_minor_ticks
                           }
        self.track_props['log_major_dash'] = self.linedash2str(tmp.xgrid.grid_line_dash)
        self.track_props['log_minor_dash'] = self.linedash2str(tmp.xgrid.minor_grid_line_dash)
        self.track_props['z_major_dash'] = self.linedash2str(tmp.ygrid.grid_line_dash)
        self.track_props['z_minor_dash'] = self.linedash2str(tmp.ygrid.minor_grid_line_dash)


    def linedash2str(self, linedash):
        dash = 'solid'
        for (key, value) in DashPattern._dash_patterns.items():
            if value==linedash:
                dash = key
                break
        return dash

    def apply_track_props(self, track):
        track.update(plot_width=self.track_props['plot_width'])
        track.update(background_fill_color=self.track_props['background_fill_color'])
        track.xgrid.visible = self.track_props['log_major_visible']
        if self.track_props['log_major_visible']:
            track.xgrid.grid_line_color = self.track_props['log_major_color']
        else:
            track.xgrid.grid_line_color = None
        track.xaxis[0].ticker.desired_num_ticks=self.track_props['log_major_num']+1
        track.xgrid.grid_line_width = self.track_props['log_major_width']
        track.xgrid.grid_line_dash = self.track_props['log_major_dash']
        if self.track_props['log_minor_visible']:
            track.xgrid.minor_grid_line_color = track.xgrid.grid_line_color
        else:
            track.xgrid.minor_grid_line_color = None
        track.xaxis[0].ticker.num_minor_ticks=self.track_props['log_minor_num']
        track.xgrid.minor_grid_line_width = self.track_props['log_minor_width']
        track.xgrid.minor_grid_line_dash = self.track_props['log_minor_dash']

        track.ygrid.visible = self.track_props['z_major_visible']
        if self.track_props['z_major_visible']:
            track.ygrid.grid_line_color = self.track_props['z_major_color']
        else:
            track.ygrid.grid_line_color = None
        track.yaxis[0].ticker.desired_num_ticks=self.track_props['z_major_num']+1
        track.ygrid.grid_line_width = self.track_props['z_major_width']
        track.ygrid.grid_line_dash = self.track_props['z_major_dash']
        if self.track_props['z_minor_visible']:
            track.ygrid.minor_grid_line_color = track.ygrid.grid_line_color
        else:
            track.ygrid.minor_grid_line_color = None
        track.yaxis[0].ticker.num_minor_ticks=self.track_props['z_minor_num']
        track.ygrid.minor_grid_line_width = self.track_props['z_minor_width']
        track.ygrid.minor_grid_line_dash = self.track_props['z_minor_dash']

class Crossplot:
    def __init__(self, well, lognms):
        self.well = well
        self.layout = None
        self.bubblepoints = None
        self.selectedstyle = None
        self.unselectedstyle = None
        self.colorbarmapper = None
        self.colormapper = None
        self.sizemapper = None
        self.regression = None
        self.colorbar = None
        self.xhist = None
        self.yhist = None
        self.props = {}
        self._init_props()
        self._initui(lognms)
        self.setLogs(lognms)

    def _initui(self, lognms):
        self.xplotfig = bp.figure(sizing_mode='stretch_both',
                                  tools='hover, pan, box_zoom, lasso_select, box_select, reset',
                                  active_drag='box_zoom',
                                  toolbar_location='above',
                                  title='vs',
                                  min_width=300,
                                  min_height=300,
                                  x_range=(0,10), y_range=(0,10))
        self.xplotfig.title.text_font_size = '12pt'
        self.xplotfig.on_event(Reset, self.resetCB)
        self.xhistfig = bp.figure(plot_height=100,
                                  sizing_mode='stretch_width',
                                  toolbar_location=None,
                                  x_range=self.xplotfig.x_range,
                                  y_range=(0,10))
        self.yhistfig = bp.figure(plot_width=100,
                                  sizing_mode='stretch_height',
                                  toolbar_location=None,
                                  y_range=self.xplotfig.y_range,
                                  y_axis_location='right',
                                  x_range=(0,10))
        self.yhistfig.xaxis.major_label_orientation='vertical'
        self.colorbarfig = bp.figure(plot_width=self.yhistfig.plot_width,
                                     plot_height=self.yhistfig.plot_height,
                                     toolbar_location=None,
                                     title_location='right',
                                     outline_line_alpha=0.0)
        self.colorbarfig.title.text_font_size = '10pt'
        self.colorbarfig.title.align='center'
        self.colorbarfig.title.text_font_style='normal'

        limits = self.well.getLogLimits(lognms[0])
        self.colorbarmapper = linear_cmap(field_name=lognms[0],
                                          palette=bpal.all_palettes[self.props['ColorMap']][256],
                                          low=limits[0], high=limits[-1])

        self.colorbar = bm.ColorBar(color_mapper=self.colorbarmapper['transform'],
                                    **self.props['ColorBar'], location=(0,0))
        self.colorbarfig.add_layout(self.colorbar, 'center')
        self.colorbarfig.visible = False

        self.setLogs(lognms)
        self.layout = bl.gridplot([[self.xplotfig, self.colorbarfig, self.yhistfig],[self.xhistfig, None, None]],
                                  merge_tools=False)
    def _init_props(self):
        self.props = {'Histograms': {'Bins': 40, 'Fill': 'darkgray'},
                      'X log': {'line_color': 'darkorange', 'line_width': 2},
                      'Y log': {'line_color': 'royalblue', 'line_width': 2},
                      'ColorMap': 'Viridis',
                      'SizeMap': (6, 25),
                      'Bubbleplot':{'line_alpha': 0.6, 'fill_alpha': 0.6,
                                    'size': 8,
                                    'line_color': 'gray', 'fill_color': 'gray'
                                    },
                      'Selected': {'size': 8, 'line_color': 'red', 'fill_color': 'red',
                                   'line_alpha': 0.6, 'fill_alpha': 0.6},
                      'Unselected': {'size': 4, 'line_alpha': 0.2, 'fill_alpha': 0.2,
                                    'line_color': 'gray', 'fill_color': 'gray'},
                      'Regression': {'Standard': {'line_color': 'navy', 'line_width': 3},
                                     'Robust': {'line_color': 'cornflowerblue', 'line_width': 3}
                                    },
                      'Legend': {},
                      'ColorBar': {'width': 15, 'label_standoff': 6}
                      }

    def resetCB(self, ev):
        if isinstance(ev, Reset):
            xlognm = self.xplotfig.xaxis.axis_label
            ylognm = self.xplotfig.yaxis.axis_label
            limits = self.well.getLogLimits(xlognm)
            self.xplotfig.x_range.start=limits[0]
            self.xplotfig.x_range.end=limits[-1]
            limits = self.well.getLogLimits(ylognm)
            self.xplotfig.y_range.start=limits[0]
            self.xplotfig.y_range.end=limits[-1]

    def updateTitle(self):
        text = '%s vs %s' % (self.xplotfig.yaxis.axis_label, self.xplotfig.xaxis.axis_label)
        self.xplotfig.title.update(text=text)

    def updateTooltips(self):
        xlognm = self.xplotfig.xaxis.axis_label
        ylognm = self.xplotfig.yaxis.axis_label
        zlognm = None
        if not xlognm or not ylognm:
            return
        if self.colormapper:
          zlognm = self.colormapper['field']
        if self.sizemapper:
          zlognm = self.sizemapper['field']

        if zlognm:
          self.xplotfig.tools[0].tooltips = [
                                              ( zlognm, '@{%s}' % zlognm ),
                                              ( xlognm, '$x' ),
                                              ( ylognm, """$y
                                              <style>
                                                  .bk-tooltip>div:not(:first-child) {display:none;}
                                              </style>""")
                                            ]
        else :
          self.xplotfig.tools[0].tooltips = [
                                              ( xlognm, '$x' ),
                                              ( ylognm, """$y
                                              <style>
                                                  .bk-tooltip>div:not(:first-child) {display:none;}
                                              </style>""")
                                            ]


    def setLogs(self, lognms):
        if len(lognms)<2:
            return
        if not self.bubblepoints:
            self.bubblepoints = self.xplotfig.circle(lognms[0], lognms[1],
                                                     **self.props['Bubbleplot'],
                                                     source=self.well.logdata,
                                                     view=self.well.logdataview)
            self.selectedstyle = bm.Circle(**self.props['Selected'])
            self.unselectedstyle = bm.Circle(**self.props['Unselected'])
            self.bubblepoints.selection_glyph = self.selectedstyle
            self.bubblepoints.nonselection_glyph = self.unselectedstyle
            cds = bm.ColumnDataSource(data={'x': [0, 1], 'ylr': [0, 1], 'yrs': [0, 1]})
            ys = [0, 1]
            self.regression = {'Standard': self.xplotfig.line('x', 'ylr',
                                                    **self.props['Regression']['Standard'],
                                                    visible=False, source=cds),
                               'Robust': self.xplotfig.line('x', 'yrs',
                                                  **self.props['Regression']['Robust'],
                                                  visible=False, source=cds),
                               'Source': cds
                              }
            self.legend = bm.Legend(items=[('Standard', [self.regression['Standard']]),
                                           ('Robust', [self.regression['Robust']])],
                                    location='top_right', visible=False)
            self.xplotfig.add_layout(self.legend)

        self.set_xlog(lognms[0])
        self.set_ylog(lognms[1])

    def set_xlog(self, lognm):
        self.xplotfig.xaxis.update(axis_label=lognm)
        self.bubblepoints.glyph.update(x=lognm)
        limits = self.well.getLogLimits(lognm)
        self.xplotfig.x_range.start=limits[0]
        self.xplotfig.x_range.end=limits[-1]
        self.set_xhistogram(lognm)
        self.updateTitle()
        self.updateTooltips()

    def set_ylog(self, lognm):
        self.xplotfig.yaxis.update(axis_label=lognm)
        self.bubblepoints.glyph.update(y=lognm)
        limits = self.well.getLogLimits(lognm)
        self.xplotfig.y_range.start=limits[0]
        self.xplotfig.y_range.end=limits[-1]
        self.set_yhistogram(lognm)
        self.updateTitle()
        self.updateTooltips()

    def set_xhistogram(self, lognm):
        log = np.array(self.well.logdata.data.get(lognm))
        xhist, xedges = np.histogram(log[np.isfinite(log)], bins=self.props['Histograms']['Bins'])
        xmax = max(xhist)*1.1
        self.xhistfig.y_range.update(start=0, end=xmax)
        if not self.xhist:
            self.xhist = self.xhistfig.quad(bottom=0, left=xedges[:-1], right=xedges[1:], top=xhist,
                                            fill_color=self.props['Histograms']['Fill'],
                                            **self.props['X log'])
        else:
            self.xhist.data_source.data.update(left=xedges[:-1], right=xedges[1:], top=xhist)

    def set_yhistogram(self, lognm):
        log = self.well.logdata.data.get(lognm)
        yhist, yedges = np.histogram(log[np.isfinite(log)], bins=self.props['Histograms']['Bins'])
        ymax = max(yhist)*1.1
        self.yhistfig.x_range.update(start=ymax, end=0)
        if not self.yhist:
            self.yhist = self.yhistfig.quad(left=0, bottom=yedges[:-1], right=yhist, top=yedges[1:],
                                            fill_color=self.props['Histograms']['Fill'],
                                            **self.props['Y log'])
        else:
            self.yhist.data_source.data.update( bottom=yedges[:-1], right=yhist, top=yedges[1:])

    def set_colormap(self, cmap):
        self.props['ColorMap'] = cmap
        if self.colormapper:
            self.set_colorlog(self.colormapper['field'])

    def set_sizemap(self, szmap):
        self.props['SizeMap'] = szmap
        if self.sizemapper:
            self.set_sizelog(self.sizemapper['field'])


    def set_colorlog(self, lognm):
        if lognm=='None':
            self.colormapper = None
            bpsettings = {}
            bpsettings.update(self.props['Bubbleplot'])
            if self.sizemapper:
                bpsettings.pop('size', None)
            self.bubblepoints.glyph.update(**bpsettings)
            if self.colorbar:
                self.colorbarfig.visible = False
            self.updateTooltips()
            return
        limits = self.well.getLogLimits(lognm)
        self.colormapper = linear_cmap(field_name=lognm,
                                       palette=bpal.all_palettes[self.props['ColorMap']][256],
                                       low=limits[0], high=limits[-1])
        self.bubblepoints.glyph.update(line_color=self.colormapper, fill_color=self.colormapper)
        self.colorbarmapper.update(field=lognm)
        self.colorbarmapper['transform'].update(low=limits[0], high=limits[-1],
                                                palette=bpal.all_palettes[self.props['ColorMap']][256])
        self.colorbarfig.title.text = lognm
        self.colorbarfig.visible=True
        self.updateTooltips()

    def set_sizelog(self, lognm):
        if lognm=='None':
            self.sizemapper = None
            self.bubblepoints.glyph.update(size=self.props['Bubbleplot']['size'])
            self.updateTooltips()
            return
        limits = self.well.getLogLimits(lognm)
        self.sizemapper = {'field': lognm,
                           'transform': bm.LinearInterpolator(x=[limits[0], limits[-1]],
                                                              y=self.props['SizeMap'])
                          }
        self.bubblepoints.glyph.update(size=self.sizemapper)
        self.updateTooltips()

    def show_regression(self, show, selectedonly=False):
        if show:
            self.set_regression(selectedonly)
        self.regression['Standard'].visible = show
        self.regression['Robust'].visible = show
        self.legend.visible = show

    def set_regression(self, selectedonly=False):
        xlognm = self.xplotfig.xaxis.axis_label
        ylognm = self.xplotfig.yaxis.axis_label
        xs = None
        ys = None
        if selectedonly and len(self.well.logdata.selected.indices)>0:
            xs = np.array(self.well.logdata.data.get(xlognm))[self.well.logdata.selected.indices]
            ys = np.array(self.well.logdata.data.get(ylognm))[self.well.logdata.selected.indices]
        else:
            xs = np.array(self.well.logdata.data.get(xlognm))[self.well.logdataviewidx]
            ys = np.array(self.well.logdata.data.get(ylognm))[self.well.logdataviewidx]
        idx = np.logical_and(np.isfinite(xs), np.isfinite(ys))
        xuse = xs[idx]
        yuse = ys[idx]
        lr = linear_model.LinearRegression()
        X = xuse.reshape(-1,1)
        lr.fit(X, yuse)
        lbl = 'Standard: {} = {} * {:.4f} + {:.4f} (R\u00B2: {:.2f})'.format(ylognm, xlognm, lr.coef_[0], 
						    lr.intercept_, lr.score(X, yuse))
        self.legend.items[0].label = value(lbl)
        xpl = np.array([np.nanmin(xs), np.nanmax(xs)])
        ylr = xpl * lr.coef_[0] + lr.intercept_
        rs = linear_model.RANSACRegressor(min_samples=0.9)
        rs.fit(X, yuse)
        yrs = xpl * rs.estimator_.coef_[0] + rs.estimator_.intercept_
        self.regression['Source'].update(data={'x': xpl, 'ylr':ylr, 'yrs': yrs})
        lbl = 'Robust: {} = {} * {:.4f} + {:.4f} (R\u00B2: {:.2f})'.format(ylognm, xlognm, rs.estimator_.coef_[0],
                                                         rs.estimator_.intercept_, rs.score(X,yuse))
        self.legend.items[1].label = value(lbl)




def update_xlog(attr, old, new, track, controls, xplot):
    track.updateLog(new, 'xlog')
    xplot.set_xlog(new)
    xplot.show_regression(len(controls.fields['regression'].active)!=0)

def update_ylog(attr, old, new, track, controls, xplot):
    track.updateLog(new, 'ylog')
    xplot.set_ylog(new)
    xplot.show_regression(len(controls.fields['regression'].active)!=0)

def update_colorlog(attr, old, new, track, controls, xplot):
    if new=='None':
        track.showLog('colorlog', False)
        controls.fields['colormap'].visible = False
    else:
        track.updateLog(new, 'colorlog')
        track.showLog('colorlog', True)
        controls.fields['colormap'].visible = True
    xplot.set_colorlog(new)

def update_sizelog(attr, old, new, track, controls, xplot):
    if new=='None':
        track.showLog('sizelog', False)
        controls.fields['sizemap'].visible = False
    else:
        track.updateLog(new, 'sizelog')
        track.showLog('sizelog', True)
        controls.fields['sizemap'].visible = True
    xplot.set_sizelog(new)

def update_colormap(attr, old, new, xplot):
    xplot.set_colormap(new)

def update_sizemap(attr, old, new, xplot):
    xplot.set_sizemap(new)

def update_markers(attr, old, new, track, controls):
    show = len(controls.fields['markers'].active)!=0
    track.show_markers(show)

def update_regression(attr, old, new, controls, xplot):
    show = len(controls.fields['regression'].active)!=0
    selonly = len(controls.fields['selected'].active)!=0
    xplot.show_regression(show, selonly)

def view_change(attr, old, new, controls, xplot):
    show = len(controls.fields['regression'].active)!=0 and len(new)!=0
    selonly = len(controls.fields['selected'].active)!=0
    xplot.show_regression(show, selonly)

def sel_change(attr, old, new, controls, xplot):
    show = len(controls.fields['regression'].active)!=0
    selonly = len(controls.fields['selected'].active)!=0
    xplot.show_regression(show, selonly)

def crossplot_app(doc):
    global survargs, wellnm, layout, welllogs
    well = odb.Well(wellnm, args=survargs)
    depths = well.depthRange()
    if welllogs == '0':
      welllogs = well.getLogIdxStr()
    data = well.getLogs(welllogs)
    lognms = well.getLogNames()

    xplcontrols = CrossplotControls(well)
    xpllogs = CrossplotLogTracks(well, lognms[1:3], 200)
    xplplots = Crossplot(well, lognms[1:3])
    xplcontrols.fields['xlog'].update(value=lognms[1])
    xplcontrols.fields['ylog'].update(value=lognms[2])
    xplcontrols.fields['sizelog'].update(value='None')
    xplcontrols.fields['colorlog'].update(value='None')
    xplcontrols.fields['colormap'].update(value=xplplots.props['ColorMap'])
    xplcontrols.fields['colormap'].visible = False
    xplcontrols.fields['sizemap'].update(value=xplplots.props['SizeMap'])
    xplcontrols.fields['sizemap'].visible = False
    xplcontrols.fields['xlog'].on_change('value', partial(update_xlog,
                                                          track=xpllogs,
                                                          controls=xplcontrols,
                                                          xplot=xplplots))
    xplcontrols.fields['ylog'].on_change('value', partial(update_ylog,
                                                          track=xpllogs,
                                                          controls=xplcontrols,
                                                          xplot=xplplots))
    xplcontrols.fields['colorlog'].on_change('value', partial(update_colorlog,
                                                              track=xpllogs,
                                                              controls=xplcontrols,
                                                              xplot=xplplots))
    xplcontrols.fields['colormap'].on_change('value', partial(update_colormap,
                                                              xplot=xplplots))
    xplcontrols.fields['sizelog'].on_change('value', partial(update_sizelog,
                                                             track=xpllogs,
                                                             controls=xplcontrols,
                                                             xplot=xplplots))
    xplcontrols.fields['sizemap'].on_change('value', partial(update_sizemap,
                                                             xplot=xplplots))

    xplcontrols.fields['markers'].on_change('active', partial(update_markers,
                                                              track=xpllogs,
                                                              controls=xplcontrols))
    xplcontrols.fields['regression'].on_change('active', partial(update_regression,
                                                                 controls=xplcontrols,
                                                                 xplot=xplplots))
    xplcontrols.fields['selected'].on_change('active', partial(update_regression,
                                                               controls=xplcontrols,
                                                               xplot=xplplots))

    well.logdataview.on_change('filters', partial(view_change, controls=xplcontrols,
                                                  xplot=xplplots))
    well.logdata.selected.on_change('indices', partial(sel_change, controls=xplcontrols,
                                               xplot=xplplots))

    layout = bl.row( xplcontrols.layout, xpllogs.layout, xplplots.layout, sizing_mode='stretch_both' )

    doc.add_root(layout)
    doc.title = "Crossplot well logs"


def main():
  global survargs, wellnm, welllogs

  survargs = {'dtectdata': ['/mnt/Data/seismic/ODData'], 'survey': ['F3_Demo_2020']}
  wellnm = 'F03-2'
  welllogs = '0,1,2'

  server = Server({'/' : crossplot_app})
  server.start()
  server.io_loop.add_callback(server.show, "/")
  server.io_loop.start()

if __name__ == "__main__":
    main()
