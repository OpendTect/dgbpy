import odpy.wellman as odwm
import numpy as np
from functools import partial
from dgbpy.bokehcore import *
import logging
import odpy.common as odcommon
from well_data import WellInfo, WellCrossplotData
from bokeh.models import Legend, LegendItem
from bokeh.palettes import Category20_20 as palette
import itertools

odcommon.proclog_logger = logging.getLogger('bokeh.bokeh_machine_learning.main')
odcommon.proclog_logger.setLevel( 'DEBUG' )

colors = itertools.cycle(palette)

class MulitWellSelector:
	def __init__(self, wellinfo):
		self.wellinfo = wellinfo
		self.apply_but = Button(label='Apply', button_type='success')
		self.well_select = MultiSelect(title='Select wells', options=self.wellinfo.names())

	def select_wells(self, wellnms):
		self.well_select.update(value=wellnms)

	def get_controls(self):
		controls = column(self.apply_but, self.well_select)
		return controls

class MultiWellLogSelector:
	def __init__(self, wellinfo, titles=['Select log'], common=True, withdepth=True):
		self.wellinfo = wellinfo
		self.loginfo = {}
		self.common = common
		self.withdepth = withdepth
		self.log_select = {}
		for title in titles:
			self.log_select[title] = Select(title=title)

		self.select_wells([self.wellinfo.names()[0]])

	def select_wells(self, wellnms):
		sellogs = None
		if self.common:
			sellogs = self.wellinfo.get_common_lognames(wellnms)
		else:
			sellogs = self.wellinfo.get_unique_lognames(wellnms)
		if self.withdepth:
			sellogs.insert(0,'depth')
		for log_select in self.log_select.values():
			log_select.options = sellogs
			if not log_select.value:
				log_select.value = log_select.options[0]

	def get_controls(self):
		controls = column(list(self.log_select.values()))
		return controls

class ColorSelector:
	def __init__(self, wellinfo, common=True, withdepth=True, withwell=True):
		self.wellinfo = wellinfo
		self.common = common
		self.withdepth = withdepth
		self.withwell = withwell
		self.color_select = Select(title='Color by')
		self.select_wells([self.wellinfo.names()[0]])

	def select_wells(self, wellnms):
		sellogs = []
		# if self.common:
		# 	sellogs = self.wellinfo.get_common_lognames(wellnms)
		# else:
		# 	sellogs = self.wellinfo.get_unique_lognames(wellnms)
		# if self.withdepth:
		# 	sellogs.insert(0,'depth')
		
		sellogs.insert(0, 'None')
		if self.withwell:
			sellogs.insert(0,'Well')

		self.color_select.update(options=sellogs)
		if not self.color_select.value:
			self.color_select.update(value=sellogs[0])

	def get_controls(self):
		controls = column(self.color_select)
		return controls

class DepthRangeSelector:
	def __init__(self, wellinfo):
		self.wd = wellinfo
		self.selmarkers = []
		self.slider = RangeSlider(step=1, title="Depth Range")
		self.depthtab = TabPanel(child=self.slider, title="Depth Range")
		self.topmarker = Select(title="Top")
		self.topoffset = Spinner(title="Above", low=-500, high=500, step=1, value=0, width=80)
		self.botmarker = Select(title="Base")
		self.botoffset = Spinner(title="Below", low=-500, high=500, step=1, value=0, width=80)
		marker_controls = column(	row(self.topmarker, self.topoffset), 
									row(self.botmarker, self.botoffset))
		self.markertab = TabPanel(child=marker_controls, title="Marker Range")
		self.tabs = Tabs(tabs=[self.depthtab,self.markertab])
		self.select_wells([self.wd.names()[0]])
		self.topmarker.on_change("value", self.on_topmarker_chg)
		self.botmarker.on_change("value", self.on_botmarker_chg)

	def select_wells(self, wellnms):
		self.selmarkers = self.wd.get_common_markernames(wellnms)
		self.selmarkers.insert(0, "Start of Data")
		self.selmarkers.append("End of Data")
		self.topmarker.options = self.selmarkers
		self.botmarker.options = self.selmarkers
		if not self.topmarker.value or self.topmarker.value not in self.selmarkers:
			self.topmarker.value = self.selmarkers[0]
		if not self.botmarker.value or self.botmarker.value not in self.selmarkers:
			self.botmarker.value = self.selmarkers[-1]
		rg = self.wd.get_depthrange(wellnms)
		self.slider.update(start=rg[0], end=rg[1], value=[rg[0], rg[1]])

	def on_topmarker_chg(self, attr, old, new):
		idx = self.selmarkers.index(new)
		self.botmarker.options = self.selmarkers[idx:] 

	def on_botmarker_chg(self, attr, old, new):
		idx = self.selmarkers.index(new)
		if idx<len(self.selmarkers):
			idx += 1
		self.topmarker.options = self.selmarkers[:idx] 

	def get_controls(self):
		return self.tabs


class MultiWellCrossPlot:
	def __init__(self):
		self.wd = WellCrossplotData()
		self.well_select = MulitWellSelector(self.wd.wellinfo)
		self.log_select = MultiWellLogSelector(self.wd.wellinfo, titles=['Crossplot X Log', 'Crossplot Y log'])
		self.color_select = ColorSelector(self.wd.wellinfo)
		self.depth_select = DepthRangeSelector(self.wd.wellinfo)
		self.xplotfig = figure(toolbar_location='right', title='Crossplot')
		self.xplot = {}
		self.well_select.apply_but.on_click(self.on_apply)
		self.well_select.well_select.on_change("value", self.on_well_select)
		self.color_select.color_select.on_change("value", self.on_color_select)
		self.depth_select.slider.on_change("value", self.on_depthfilter)
		self.depth_select.tabs.on_change("active", self.on_depthsel_chg)
		self._logsel = [None, None]

	def select_wells(self, wellnms):
		self.well_select.select_wells(wellnms)

	def on_well_select(self, attr, old, new):
		self.log_select.select_wells(new)
		self.depth_select.select_wells(new)
		self.color_select.select_wells(new)

	def on_color_select(self, attr, old, new):
		wellnms = self.well_select.well_select.value
		if self.color_select.color_select.value=='None':
			self.xplotfig.legend.items = []
			for well in wellnms:
				if well in self.xplot:
					self.xplot[well].glyph.update(fill_color="blue", line_color="blue")
		elif self.color_select.color_select.value=='Well':
			for well in wellnms:
				usecolor = next(colors)
				if well in self.xplot:
					self.xplot[well].glyph.update(fill_color=usecolor, line_color=usecolor)
	
			self.xplotfig.legend.items = [LegendItem(label=well, renderers=[self.xplot[well]]) for well in wellnms]

	def hideall(self):
		for well in self.xplot:
			self.xplot[well].visible = False
			self.xplotfig.legend.items = []

	def on_apply(self):	
		wellnms = self.well_select.well_select.value
		xlog = list(self.log_select.log_select.values())[0].value
		ylog = list(self.log_select.log_select.values())[1].value
		self.wd.get_data(wellnms, xlog, ylog)
		self.hideall()
		for well in wellnms:
			if well in self.xplot:
				self.xplot[well].visible = True
			else:
				usecolor = "blue" if self.color_select.color_select.value=='None' else next(colors)
				self.xplot[well] = self.xplotfig.circle("xlog", "ylog", source = self.wd.cdsdata[well], view=self.wd.cdsviews[well], 
										legend_label=well, color=usecolor)

		if self.color_select.color_select.value=='Well':
			self.xplotfig.legend.items = [LegendItem(label=well, renderers=[self.xplot[well]]) for well in wellnms]

		ds = self.depth_select
		if ds.tabs.active==0:
			vrg = self.depth_select.slider.value
			self.wd.filter_depth_range(vrg[0], vrg[1])
		else:
			self.wd.filter_marker_range(ds.topmarker.value, ds.botmarker.value, ds.topoffset.value, ds.botoffset.value)

		self.update_axes()

	def on_depthfilter(self, attr, old, new):
		self.wd.filter_depth_range(new[0], new[1])

	def on_depthsel_chg(self, attr, old, new):
		if new==0:
			rg = self.depth_select.slider.value
			self.wd.filter_depth_range(rg[0], rg[1])
		else:
			ds = self.depth_select
			self.wd.filter_marker_range(ds.topmarker.value, ds.botmarker.value, ds.topoffset.value, ds.botoffset.value)

	def get_controls(self):
		controls = column(	self.well_select.get_controls(), self.log_select.get_controls(), self.color_select.get_controls(),
							self.depth_select.get_controls())
		self.xplotfig.sizing_mode = 'stretch_both'
		self.dashboard = row(controls, self.xplotfig)
		self.dashboard.sizing_mode = 'stretch_both'
		return self.dashboard

	def update_axes(self):
			self.xplotfig.xaxis.axis_label = self.wd.logsel[0]
			self.xplotfig.yaxis.axis_label = self.wd.logsel[1]

	def get_log_selection(self):
		return self.wd.logsel
