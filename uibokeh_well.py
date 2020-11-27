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
import pandas as pd
import numpy as np
import random
from odpy.ranges import niceRange, niceNumber

class Well:
    def __init__(self,wellname):
        self.wellname = wellname
        self.track = None
        self.markers = None
        self.logcache = None
        self.logdata = None
        self.logdataview = None
        self.logdataviewidx = None
        self.limits = {}

    def getLogNames(self):
        if not self.logdata:
            return odwm.getLogNames(self.wellname)
        else:
            return self.logdata.column_names[1:]

    def getLogsFromFile(self, filenm, undefval=1e30):
        ld = pd.read_csv(filenm, delimiter='\t')
        ld = ld.replace(to_replace=undefval, value=float('Nan'))
        self.logdata = bm.ColumnDataSource(ld)
        self.logdataviewidx = self.logdata.data['index']
        self.logdataview = bm.CDSView(source=self.logdata, filters=[])
        return self.logdata

    def getLogLimits(self, lognm):
        limits = self.limits.get(lognm)
        if not limits:
            log = None
            if not self.logdata:
                log = self.getLog(lognm).data.get(lognm)
            else:
                log = self.logdata.data.get(lognm)
            limits = niceRange(np.nanmin(log), np.nanmax(log))
            self.limits[lognm] = limits

        return limits

    def setDepthView(self, mindepth, maxdepth):
        if self.logdataview:
            self.logdataviewidx = [idx for idx, z in enumerate(self.logdata.data['MD']) if z>=mindepth and z<=maxdepth]
            self.logdataview.filters = [bm.IndexFilter(indices=self.logdataviewidx)]

    def setLogLimits(self, lognm, left, right):
        self.limits[lognm] = [left, right]

    def getLog(self, lognm):
        if (not self.logcache) or (lognm not in self.logcache):
            (depths,logvals) = odwm.getLog(self.wellname, lognm)
            if not self.logcache:
               self.logcache = {}
            self.logcache[lognm] = bm.ColumnDataSource({'depth': depths, lognm: logvals})
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
        if self.logdata:
            return self.getLogLimits('MD')
        elif self.logcache:
            depthrange = None
            first = True
            for (key, item) in self.logcache.items():
                zlog = item.data['depth']
                if first:
                    first = False
                    depthrange = [zlog[0], zlog[-1]]
                else:
                    depthrange = [min(zlog[0],depthrange[0]), max(zlog[-1],depthrange[-1])]

            return depthrange
        else:
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
        tracktabs = self.tracklayout.children[0].children[0]
        tracktabs.active = 1
        if len(deflogs)<=1 :
          tracktabs.active = 0


    def _addtrack(self, copytrack=None):

        if self.withui:
            addbutton = bm.Button(label='+', button_type='success', width_policy='min')
            addbutton.on_event(ButtonClick, self._addbuttonCB)
            rmvbutton = bm.Button(label='-', button_type='warning', width_policy='min')
            rmvbutton.on_event(ButtonClick, self._rmvbuttonCB)
            syncbutton = bm.CheckboxGroup(labels=['Sync'], active=[0], width_policy='min')
            syncbutton.on_change('active', self._syncbuttonCB)
            newtrack = LogTrack(self.well, self.width, withui=self.withui)
            if copytrack:
                newtrack.log_props = copytrack.log_props.copy()
                newtrack.track_props = copytrack.track_props.copy()
                master_trackbox = self.tracklayout.children[0]
                master_trackfig = self.tracks[master_trackbox.name].track
                newtrack.track.y_range.start = master_trackfig.y_range.start
                newtrack.track.y_range.end = master_trackfig.y_range.end
                newtrack.track.y_range.bounds = "auto"

                master_trackfig.y_range.js_on_change('start',
                          bm.CustomJS(args=dict(rg=newtrack.track.y_range,sb=syncbutton),
                                      code="if (sb.active.length===1) rg.start = this.start;"))
                master_trackfig.y_range.js_on_change('end',
                          bm.CustomJS(args=dict(rg=newtrack.track.y_range,sb=syncbutton),
                                      code="if (sb.active.length===1) rg.end = this.end;"))
                newtrack.apply_track_props()
                logs = copytrack.log_select.value
                newtrack.addLogs(logs)
                markers = copytrack.marker_select.value
                newtrack.addMarkers(markers)
            else:
                newtrack.addLogs(self.deflogs)
                rmvbutton.visible = False
                syncbutton.visible = False

            buttons = bl.row([addbutton,rmvbutton,syncbutton], align='center')
            trackbox = bl.column([newtrack.display(), buttons], name=str(self.trackid), tags=[addbutton.id, rmvbutton.id])
            trackbox.children[0].active = 1
            self.tracks[str(self.trackid)] = newtrack
            self.trackid += 1
            if self.tracklayout:
                self.tracklayout.children.append(trackbox)
                self._syncbuttonCB('active', 0, 0)
            else:
                self.tracklayout = bl.row([trackbox], sizing_mode='stretch_height')
                rmvbutton.visible = False
                syncbutton.visible = False
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


    def _rmvbuttonCB(self, event ):
        evid = event._model_id
        for idx in range(1, len(self.tracklayout.children)):
            trackbox = self.tracklayout.children[idx]
            if evid in trackbox.tags:
                self.tracks.pop(trackbox.name)
                self.tracklayout.children.pop(idx)
                break

    def _syncbuttonCB(self, attr, old, new):
        numtracks = len(self.tracklayout.children)
        for idx in range(1, numtracks):
          trackbox = self.tracklayout.children[idx]
          trackid = trackbox.name
          syncbutton = trackbox.children[-1].children[-1]
          logtrack = self.tracks[trackid]
          trackfig = logtrack.track
          if len(syncbutton.active)>0 and idx>0:
            trackfig.tools = []
          else:
            trackfig.tools = logtrack.default_tools
            trackfig.toolbar.active_scroll = logtrack.active_scroll



class LinePropertyWidget:
    def __init__(self):
        self.fields = {'width': bm.Spinner(title='Line Width:',low=0,high=5, step=1,
                                           width_policy='min'),
                       'color': bm.ColorPicker(title='Line Color:', width_policy='min'),
                       'dash':  bm.Select(title='Line Draw Style:', options=list(bce.DashPattern),
                                          width_policy='min')
                       }
        self.layout = bl.row(list(self.fields.values()))

class GridPropertyWidget:
    def __init__(self, title, withsteps=True):
        line_props = LinePropertyWidget()
        field_title = bm.Div(text=title)
        self.fields = {'visible': bm.CheckboxGroup(labels=[''], inline= True, active=[0], width_policy='min'),
                       'steps': bm.Spinner(title='Steps:',low=0,high=5, step=1)
                      }
        self.fields.update(line_props.fields)
        self.fields['visible'].on_change('active',self.visible)
        self.layout = bl.column(bl.row(self.fields['visible'],field_title),
                                self.fields['steps'],
                                line_props.layout
                                )

    def visible(self, attr, old, new):
        disable = False
        if len(new)==0:
            disable = True
        for (key, field) in self.fields.items():
            if key is 'visible':
                continue
            field.disabled = disable

class LogTrack:
    def __init__(self, well, width, withui=False):
        self.well = well
        self.width = width
        self.withui = withui
        self.tabs = None
        self.tracklayout = None
        self.tooltips = [("Depth", " $y"),
                         ("Log","$x"),
                        ]
        self.track_props = {}
        self.track_fields = {}
        self._inittrackprops()
        self.log_fields = {}
        self.log_props = {}
        self._initlogprops()
#        self.shading =np.empty((height,width,4),dtype=np.uint8)
        self._inittrack()
        self._initui()
        self.default_tools = self.track.tools
        self.active_scroll = self.track.toolbar.active_scroll

    def _inittrack(self):
        depths = self.well.depthRange()
        self.track = bp.figure(title=self.well.wellname,
                             plot_width=self.track_props['plot_width'],
                             sizing_mode='stretch_height',
                             background_fill_color=self.track_props['background_fill_color'],
                             tools='ypan, reset, hover',
                             y_axis_label='Depth (mMD)',
                             tooltips=self.tooltips)
        wheelzoom = bm.WheelZoomTool(dimensions="height", maintain_focus=False)
        self.track.add_tools(wheelzoom)
        boxzoom = bm.BoxZoomTool(dimensions="height")
        self.track.add_tools(boxzoom)
        self.track.toolbar.active_scroll = wheelzoom
        self.track.toolbar.logo = None
        self.track.title.text_font_size = '14pt'
        self.track.y_range = bm.Range1d(depths[-1], depths[0],bounds="auto")
        self.track.x_range = bm.Range1d(0, 10)
        self.track.xaxis.visible = False
        self.apply_track_props()


    def _inittrackprops(self):
        tmp = bp.figure()
        self.track_props = {'plot_width': self.width,
                            'background_fill_color': tmp.background_fill_color,
                            'log_major_visible': tmp.xgrid.visible,
                            'log_major_num': tmp.xaxis[0].ticker.desired_num_ticks-1,
                            'log_major_width': tmp.xgrid.grid_line_width,
                            'log_major_color': tmp.xgrid.grid_line_color,
                            'log_minor_visible': tmp.xgrid.minor_grid_line_color is not None,
                            'log_minor_width': tmp.xgrid.minor_grid_line_width,
                            'log_minor_color': tmp.xgrid.grid_line_color,
                            'log_minor_num': tmp.xaxis[0].ticker.num_minor_ticks,
                            'z_major_visible': tmp.ygrid.visible,
                            'z_major_num': tmp.yaxis[0].ticker.desired_num_ticks-1,
                            'z_major_width': tmp.ygrid.grid_line_width,
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

    def _initlogprops(self):
        lognames = self.well.getLogNames()
        for log in lognames:
            self.log_props[log] = {'left': None,
                                   'right': None,
                                   'width': 2,
                                   'color': "#%06x" % random.randint(0,0xFFFFFF),
                                   'dash': 'solid'
                                   }

    def _init_log_ui(self):
        lognames = self.well.getLogNames()
        line_props = LinePropertyWidget()
        self.log_fields= {'log': bm.Select(title='Log Display Styling:', options=lognames),
                          'left': bm.Spinner(title='Left Limit:'),
                          'right': bm.Spinner(title='Right Limit:'),
                         }
        self.log_fields.update(line_props.fields)
        self.log_fields['log'].on_change('value', self._update_log_selection)
        self.log_select.js_link('value',self.log_fields['log'], 'options')
        return bl.layout(self.log_fields['log'],
                         self.log_fields['left'],
                         self.log_fields['right'],
                         line_props.layout,
                         align=('center','start')
                        )

    def _initui(self):
        lognames = self.well.getLogNames()
        self.log_select = bm.MultiSelect(title="Logs:", options=lognames, align='center',
                                         height_policy='max')
        self.marker_select = bm.MultiSelect(title="Markers:", options=self.well.getMarkers().data['name'],
                                            align='center', height_policy='max')

        self.track_fields['plot_width'] = bm.Spinner(title='Plot Width(px):', value=self.track_props['plot_width'])
        self.track_fields['background_fill_color'] = bm.ColorPicker(title='Track Background Color:', color=self.track_props['background_fill_color'])
        major_log_grid = GridPropertyWidget('Major Log Grid')
        self.track_fields['log_major'] = major_log_grid.fields
        minor_log_grid = GridPropertyWidget('Minor Log Grid')
        self.track_fields['log_minor'] = minor_log_grid.fields
        major_z_grid = GridPropertyWidget('Major Z Grid')
        self.track_fields['z_major'] = major_z_grid.fields
        minor_z_grid = GridPropertyWidget('Minor Z Grid')
        self.track_fields['z_minor'] = minor_z_grid.fields
        self._update_track_ui()

        self.data_ui = bl.column(self.log_select,
                                 self.marker_select,
                                 self._init_log_ui(),
                                 align=('center','start')
                                )
        self.settings_ui = bl.column(self.track_fields['plot_width'],
                                     self.track_fields['background_fill_color'],
                                     major_log_grid.layout,
                                     minor_log_grid.layout,
                                     major_z_grid.layout,
                                     minor_z_grid.layout,
                                     align=('center','start')
                                    )

    def _update_track_ui(self):
        self.track_fields['plot_width'].update(value=self.track_props['plot_width'])
        self.track_fields['background_fill_color'].update(color=self.track_props['background_fill_color'])

        major_log_fields = self.track_fields['log_major']
        if self.track_props['log_major_visible']:
            major_log_fields['visible'].update(active=[0])
        else:
            major_log_fields['visible'].update(active=[])
        major_log_fields['steps'].update(value=self.track_props['log_major_num'])
        major_log_fields['width'].update(value=self.track_props['log_major_width'])
        major_log_fields['color'].update(color=self.track_props['log_major_color'])
        major_log_fields['dash'].update(value=self.track_props['log_major_dash'])

        minor_log_fields = self.track_fields['log_minor']
        if self.track_props['log_minor_visible']:
            minor_log_fields['visible'].update(active=[0])
        else:
            minor_log_fields['visible'].update(active=[])
        minor_log_fields['steps'].update(value=self.track_props['log_minor_num'])
        minor_log_fields['width'].update(value=self.track_props['log_minor_width'])
        minor_log_fields['color'].update(color=self.track_props['log_minor_color'])
        minor_log_fields['dash'].update(value=self.track_props['log_minor_dash'])

        major_z_fields = self.track_fields['z_major']
        if self.track_props['z_major_visible']:
            major_z_fields['visible'].update(active=[0])
        else:
            major_z_fields['visible'].update(active=[])
        major_z_fields['steps'].update(value=self.track_props['z_major_num'])
        major_z_fields['width'].update(value=self.track_props['z_major_width'])
        major_z_fields['color'].update(color=self.track_props['z_major_color'])
        major_z_fields['dash'].update(value=self.track_props['z_major_dash'])

        minor_z_fields = self.track_fields['z_minor']
        if self.track_props['z_minor_visible']:
            minor_z_fields['visible'].update(active=[0])
        else:
            minor_z_fields['visible'].update(active=[])
        minor_z_fields['steps'].update(value=self.track_props['z_minor_num'])
        minor_z_fields['width'].update(value=self.track_props['z_minor_width'])
        minor_z_fields['color'].update(color=self.track_props['z_minor_color'])
        minor_z_fields['dash'].update(value=self.track_props['z_minor_dash'])

    def linedash2str(self, linedash):
        dash = 'solid'
        for (key, value) in DashPattern._dash_patterns.items():
         if value==linedash:
            dash = key
            break
        return dash

    def _update_log_selection(self, attr, old, new):
        lognm = new
        left = self.log_props[lognm]['left']
        right = self.log_props[lognm]['right']
        if left is None or right is None:
            self.addLog(lognm)
        self._update_log_ui()

    def _update_log_ui(self):
        lognm = self.log_fields['log'].value
        left = self.log_props[lognm]['left']
        right = self.log_props[lognm]['right']
        nicestep = niceNumber(abs(right-left)/10)
        self.log_fields['right'].update(step=nicestep)
        self.log_fields['left'].update(step=nicestep)
        for (key, prop) in self.log_props[lognm].items():
            if key == 'color':
                self.log_fields[key].update(color=prop)
            else:
                self.log_fields[key].update(value=prop)

    def _update_log_props(self):
        lognm = self.log_fields['log'].value
        for (key, field) in self.log_fields.items():
            if key == 'color':
                self.log_props[lognm][key] = field.color
            else:
                self.log_props[lognm][key] = field.value

        start = self.log_fields['left'].value
        end = self.log_fields['right'].value
        color = self.log_fields['color'].color
        width = int(self.log_fields['width'].value)
        dash = self.log_fields['dash'].value
        self.track.extra_x_ranges[lognm].update(start=start, end=end)
        line = self.track.select(name=lognm).glyph
        line.update(line_width=width, line_color=color, line_dash=dash)
        axis = self.track.select(axis_label=lognm)
        axis.update(axis_line_color=color, major_tick_line_color=color,
                    axis_line_width=width,major_tick_line_width=width,
                    axis_line_dash=dash, ticker=(start, end))

    def _update_track_props(self):
        self.track_props['plot_width'] = self.track_fields['plot_width'].value
        self.track_props['background_fill_color'] = self.track_fields['background_fill_color'].color
        self.track_props['log_major_visible'] = self.track_fields['log_major']['visible'].active==[0]
        self.track_props['log_major_num'] = self.track_fields['log_major']['steps'].value
        self.track_props['log_major_width'] = self.track_fields['log_major']['width'].value
        self.track_props['log_major_color'] = self.track_fields['log_major']['color'].color
        self.track_props['log_major_dash'] = self.track_fields['log_major']['dash'].value
        self.track_props['log_minor_visible'] = self.track_fields['log_minor']['visible'].active==[0]
        self.track_props['log_minor_num'] = self.track_fields['log_minor']['steps'].value
        self.track_props['log_minor_width'] = self.track_fields['log_minor']['width'].value
        self.track_props['log_minor_color'] = self.track_fields['log_minor']['color'].color
        self.track_props['log_minor_dash'] = self.track_fields['log_minor']['dash'].value
        self.track_props['z_major_visible'] = self.track_fields['z_major']['visible'].active==[0]
        self.track_props['z_major_num'] = self.track_fields['z_major']['steps'].value
        self.track_props['z_major_width'] = self.track_fields['z_major']['width'].value
        self.track_props['z_major_color'] = self.track_fields['z_major']['color'].color
        self.track_props['z_major_dash'] = self.track_fields['z_major']['dash'].value
        self.track_props['z_minor_visible'] = self.track_fields['z_minor']['visible'].active==[0]
        self.track_props['z_minor_num'] = self.track_fields['z_minor']['steps'].value
        self.track_props['z_minor_width'] = self.track_fields['z_minor']['width'].value
        self.track_props['z_minor_color'] = self.track_fields['z_minor']['color'].color
        self.track_props['z_minor_dash'] = self.track_fields['z_minor']['dash'].value

    def apply_track_props(self):
        self.track.update(plot_width=self.track_props['plot_width'])
        self.track.update(background_fill_color=self.track_props['background_fill_color'])
        self.track.xgrid.visible = self.track_props['log_major_visible']
        if self.track_props['log_major_visible']:
            self.track.xgrid.grid_line_color = self.track_props['log_major_color']
        else:
            self.track.xgrid.grid_line_color = None
        self.track.xaxis[0].ticker.desired_num_ticks=self.track_props['log_major_num']+1
        self.track.xgrid.grid_line_width = self.track_props['log_major_width']
        self.track.xgrid.grid_line_dash = self.track_props['log_major_dash']
        if self.track_props['log_minor_visible']:
            self.track.xgrid.minor_grid_line_color = self.track.xgrid.grid_line_color
        else:
            self.track.xgrid.minor_grid_line_color = None
        self.track.xaxis[0].ticker.num_minor_ticks=self.track_props['log_minor_num']
        self.track.xgrid.minor_grid_line_width = self.track_props['log_minor_width']
        self.track.xgrid.minor_grid_line_dash = self.track_props['log_minor_dash']

        self.track.ygrid.visible = self.track_props['z_major_visible']
        if self.track_props['z_major_visible']:
            self.track.ygrid.grid_line_color = self.track_props['z_major_color']
        else:
            self.track.ygrid.grid_line_color = None
        self.track.yaxis[0].ticker.desired_num_ticks=self.track_props['z_major_num']+1
        self.track.ygrid.grid_line_width = self.track_props['z_major_width']
        self.track.ygrid.grid_line_dash = self.track_props['z_major_dash']
        if self.track_props['z_minor_visible']:
            self.track.ygrid.minor_grid_line_color = self.track.ygrid.grid_line_color
        else:
            self.track.ygrid.minor_grid_line_color = None
        self.track.yaxis[0].ticker.num_minor_ticks=self.track_props['z_minor_num']
        self.track.ygrid.minor_grid_line_width = self.track_props['z_minor_width']
        self.track.ygrid.minor_grid_line_dash = self.track_props['z_minor_dash']

    def _change_tab(self, attr, old, new):
        if old==0:
            self._update_log_props()
        elif old==2:
            self._update_track_props()

        if new==1:
            currange = self.track.y_range
            self._inittrack()
            self.track.update(y_range=currange)
            self.addLogs(self.log_select.value)
            self.addMarkers(self.marker_select.value)
            if self.tracklayout:
                self.tracklayout.children[0] = self.track
        elif new==0:
            self._update_log_ui()
        elif new==2:
            self._update_track_ui()


    def display(self):
        if self.withui:
          self.tracklayout = bl.column(self.track)
          tracktab = bm.Panel(child=self.tracklayout, title="Plot")
          datatab = bm.Panel(child=self.data_ui, title="Data")
          settingstab = bm.Panel(child=self.settings_ui, title="Settings")
          self.tabs = bm.Tabs(tabs=[datatab, tracktab, settingstab])
          self.tabs.on_change('active', self._change_tab)
          return self.tabs
        else:
          return self.track

    def _getLogLimits(self, lognm):
        log = self.well.getLog(lognm)
        left = self.log_props[lognm]['left']
        right = self.log_props[lognm]['right']
        limits = (left, right)
        if left is None or right is None:
            limits = niceRange(min(log.data[lognm]), max(log.data[lognm]))
            self.log_props[lognm]['left'] = limits[0]
            self.log_props[lognm]['right'] = limits[-1]
        return limits

    def addLog(self, lognm):
        log = self.well.getLog(lognm)
        limits = self._getLogLimits(lognm)
        width = self.log_props[lognm]['width']
        color = self.log_props[lognm]['color']
        dash = self.log_props[lognm]['dash']

        axis_style = {'axis_line_color': color,
                      'axis_line_width': width,
                      'axis_line_dash': dash,
                      'major_tick_line_color': color,
                      'major_tick_line_width': width,
                      'major_tick_in': 0,
                      'axis_label_standoff': -10
                     }
        log_style = {'line_color': color,
                     'line_width': width,
                     'line_dash': dash
                    }
        self.track.extra_x_ranges[lognm] = bm.Range1d(limits[0], limits[-1])
        self.track.add_layout(bm.LinearAxis(x_range_name=lognm,axis_label=lognm,ticker=limits,**axis_style), 'above')
        self.track.line(x=lognm, y='depth',x_range_name=lognm,source=log, name=lognm,  **log_style)

    def addLogs(self, logs):
        lognms = self.well.getLogNames()
        showlogs = list(set.intersection(set(logs), set(lognms)))
        for lognm in showlogs:
            self.addLog(lognm)
        if self.withui:
            self.log_select.update(value=showlogs)
            self.log_fields['log'].update(options=showlogs, value=showlogs[0])
            self._update_log_ui()

    def addMarker(self, name, depth, color):
        marker_style = {'line_color': color,
                        'line_width': 2,
                        'line_dash': 'solid'
                       }
        xr = self.track.x_range
        self.track.add_layout(bm.Arrow(end=None,x_start=xr.start,y_start=depth,x_end=xr.end,y_end=depth,**marker_style))
        self.track.add_layout(bm.Label(x=xr.start,y=depth,text=name,text_color=color,render_mode='canvas'))

    def addMarkers(self, markers=None):
        if not markers or len(markers)==0:
            return
        mrksource = self.well.getMarkers().data
        showmarkers = []
        for (name,depth,color) in zip( mrksource['name'],mrksource['depth'],mrksource['color']):
            if name in markers:
                self.addMarker(name,depth,color)
                showmarkers += [name]
        if self.withui:
          self.marker_select.update(value=showmarkers)

    def clearShading(self):
        self.shading = self.back_color

    def addLeftShade(self,log,color):
        pass
