import odpy.wellman as odwm
import numpy as np
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, MultiSelect, Select, Button, RadioGroup, Spinner
from bokeh.plotting import figure

class MulitWellSelector:
	def __init__(self):
		self._wellnms = odwm.getNames()
		self.apply_but = Button(label='Apply', button_type='success')

		self.well_select = MultiSelect(title='Select wells', width_policy='max', height_policy='fit',	options=self._wellnms)

	def get_controls(self):
		controls = column(self.apply_but, self.well_select)
		return controls

class MultiWellLogSelector:
	def __init__(self, wellnms, titles=['Select log'], common=True, withdepth=True):
		self.loginfo = {}
		self.common = common
		self.withdepth = withdepth
		self.log_select = {}
		self.set_wells(wellnms)
		for title in titles:
			self.log_select[title] = Select(title=title, width_policy='max')
		self.update_select()

	def set_wells(self, wellnms):
		for well in wellnms:
			logs = odwm.getLogNames(well)
			self.loginfo[well] = odwm.getLogNames(well)

	def get_commonlogs(self, withdepth=True):
		logset = set()
		for lognms in self.loginfo.values():
			if logset:
				logset = logset & set(lognms)
			else:
				logset.update(lognms)
		if withdepth:
			logset.add('depth')
		return sorted(logset)

	def get_uniquelogs(self, withdepth=True):
		logset = set()
		for lognms in self.loginfo.values():
			logset.update(lognms)
		if withdepth:
			logset.add('depth')
		return sorted(logset)

	def update_select(self):
		for log_select in self.log_select.values():
			if self.common:
				log_select.options = self.get_commonlogs(self.withdepth)
			else:
				log_select.options = self.get_uniquelogs(self.withdepth)
			if not log_select.value:
				log_select.value = log_select.options[0]

	def get_controls(self):
		controls = column(list(self.log_select.values()))
		return controls

	def get_selected_data(self, well):
		idxstr = ''
		data = {}
		if well in self.loginfo.keys():
			for log_select in self.log_select.values():
				log = log_select.value
				if not log:
					return data
				if log!='depth':
					if idxstr:
						idxstr += ','
					idxstr += str(self.loginfo[well].index(log))
			if idxstr:
					data = odwm.getLogs(well, idxstr)
					for ky in data:
						vals = np.array(data[ky],dtype=float)
						vals[vals==1e30] = np.nan
						data[ky] = vals
			else:
				mdrng = odwm.getTrack(well)[0]
				data['depth'] = np.arange(mdrng[0], mdrng[1], 1)
			nv = data['depth'].size
			data['well'] = [well]*nv
		return data


class MultiWellCrossPlot:
	def __init__(self):
		self.well_select = MulitWellSelector()
		self.log_select = MultiWellLogSelector([], titles=['Crossplot X Log', 'Crossplot Y log'])
		self._logdata = ColumnDataSource(data={'depth': [], 'xlog': [], 'ylog': [], 'well': []})
		self.xplot = figure(toolbar_location='right', title='Crossplot', sizing_mode='stretch_both')
		self.xplot.circle('xlog', 'ylog', source = self._logdata)
		self.well_select.apply_but.on_click(self.on_apply)
		self.well_select.well_select.on_change("value", self.on_well_select)
		self._logsel = [None, None]

	def on_well_select(self, attr, old, new):
		self.log_select.set_wells(new)
		self.log_select.update_select()

	def on_apply(self):	
			wells = self.well_select.well_select.value
			xlog = list(self.log_select.log_select.values())[0].value
			ylog = list(self.log_select.log_select.values())[1].value
			new_data = {}
			for well in wells:
				data = self.log_select.get_selected_data(well)
				if not data:
					return
				xlogky = 'depth'
				lidx = 0
				if xlog!='depth':
					lidx += 1
					xlogky = list(data.keys())[lidx]
				ylogky = 'depth'
				if ylog!='depth':
					if xlog!=ylog:
						lidx += 1
					ylogky = list(data.keys())[lidx]

				if 'depth' in new_data:
					new_data['depth'] = np.append(new_data['depth'], data['depth'])
					new_data['well'] = np.append(new_data['well'], data['well'])
					new_data['xlog'] = np.append(new_data['xlog'], data[xlogky])
					new_data['ylog'] = np.append(new_data['ylog'], data[ylogky])
				else:
					new_data['depth'] = data['depth']
					new_data['well'] = data['well']
					new_data['xlog'] = data[xlogky]
					new_data['ylog'] = data[ylogky]
					self._logsel = [xlogky, ylogky]
			self._logdata.data = new_data
			self.update_axes()

	def get_controls(self):
		controls = column(self.well_select.get_controls(), self.log_select.get_controls())
		dashboard = row(controls, self.xplot, sizing_mode='stretch_height')
		return dashboard

	def update_axes(self):
			self.xplot.xaxis.axis_label = self._logsel[0]
			self.xplot.yaxis.axis_label = self._logsel[1]

	def get_log_selection(self):
		return self._logsel