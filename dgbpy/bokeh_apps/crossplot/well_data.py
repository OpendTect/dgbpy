# __________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        Wayne Mogg
# Date:          June 2022
#
# _________________________________________________________________________
# various tools to handle well data in Bokeh apps
# 
import odpy.wellman as odwm
import numpy as np
from bokeh.models import ColumnDataSource, CDSView, BooleanFilter
import odbind as odb
from odbind.survey import Survey
from odbind.well import Well

import config as cfg


class WellInfo:
  def __init__(self):
    self.survey = Survey(odb.get_user_survey())

  def names(self):
    return Well.names(self.survey)

  def lognames(self, well):
    try:
      wellobj = Well(self.survey, well)
      if not wellobj.isok:
        return []

      return wellobj.log_names
    except:
      return []

  def markernames(self, well):
    try:
      wellobj = Well(self.survey, well)
      if not wellobj.isok:
        return []

      return wellobj.marker_names
    except:
      return []

  def markerdepths(self, well):
    try:
      wellobj = Well(self.survey, well)
      if not wellobj.isok:
        return []

      return [marker['dah'] for marker in wellobj.marker_info()]
    except:
      return []

  def markerdepth(self, well, marker):
    try:
      wellobj = Well(self.survey, well)
      if not wellobj.isok:
        return np.nan
    
      return wellobj.marker_info([marker])[0]['dah']
    except:
      return np.nan

  def depthrange(self, well):
    depthrg = [0,1000]
    idx = None
    try:
      wellobj = Well(self.survey, well)
      if wellobj.isok:
        track = wellobj.track()
        depthrg = [track['dah'][0], track['dah'][-1]]

      return depthrg
    except:
      return depthrg

  def get_depthrange(self, wellnms):
    depthrg = []
    for well in wellnms:
      rg = self.depthrange(well)
      if depthrg:
        depthrg[0] = min(rg[0], depthrg[0])
        depthrg[1] = max(rg[1], depthrg[1])
      else:
        depthrg = rg
    return depthrg

  def get_common_lognames(self, wellnms):
    logset = set()
    for well in wellnms:
      lognms = self.lognames(well)
      if logset:
        logset = logset & set(lognms)
      else:
        logset.update(lognms)
    return sorted(logset)

  def get_unique_lognames(self, wellnms):
    logset = set()
    for well in wellnms:
      lognms = self.lognames(well)
      logset.update(lognms)
    return sorted(logset)

  def logidxs(self, well, logs):
    idxs = []
    if well in self.names():
      lognms = self.lognames(well)
      for log in logs:
        try:
          idxs.append(lognms.index(log))
        except ValueError:
          pass
    return idxs

  def get_logdata(self, well, logsel):
    data = {}
    uom = []
    sel = [log for log in logsel if log!='dah']
    wellobj =Well(self.survey, well)
    if wellobj.isok and sel:
      data, uom = wellobj.logs(sel)
      logkeys = {ky: f"{ky} ({uomky})" for ky, uomky in zip(data.keys(),uom)}
      data['logkeys'] = logkeys
    else:
      mdrng = self.depthrange(well)
      data['dah'] = np.arange(mdrng[0], mdrng[-1], 1)
      data['logkeys'] = {'dah': 'depth'}
    nv = data['dah'].size
    data['well'] = [well]*nv
    return data

  def get_common_markernames(self, wellnms):
    mrkset = []
    for well in wellnms:
      mrknms = self.markernames(well)
      if mrkset:
        mrkset = [nm for nm in mrkset if nm in mrknms]
      else:
        mrkset = [nm for nm in mrknms]
    return mrkset

  def get_unique_markernames(self, wellnms):
    mrkset = set()
    for well in wellnms:
      mrknms = self.markernames(well)
      mrkset.update(mrknms)
    return list(mrkset)

class WellCrossplotData:
  def __init__(self):
    self.logsel = []
    self.wellinfo = WellInfo()
    self.cdsdata = {}
    self.cdsviews = {}

  def get_data(self, wellnms, xlog, ylog):
    if xlog=="depth":
      xlog = "dah"
    if ylog=="depth":
      ylog = "dah"
    for well in wellnms:
      data = self.wellinfo.get_logdata(well, [xlog, ylog])
      if not data:
        return

      if well in self.cdsdata:
        self.cdsdata[well].update(data={'depth': data['dah'], 'xlog': data[xlog], 'ylog': data[ylog]})
      else:
        self.cdsdata[well] = ColumnDataSource(data={'depth': data['dah'], 'xlog': data[xlog], 'ylog': data[ylog]})
        self.cdsviews[well] = CDSView(filters=[], source=self.cdsdata[well])
      logkys = data.get('logkeys', {})
      
    self.logsel = [logkys[xlog], logkys[ylog]]

  def filter_reset(self):
    for well in self.cdsviews:
      self.cdsviews[well].update(filters=[])

  def filter_depth_range(self, mindepth, maxdepth):
    for well in self.cdsdata:
      zfilter = [True if z>=mindepth and z<=maxdepth else False for z in self.cdsdata[well].data['depth']]
      self.cdsviews[well].update(filters=[BooleanFilter(zfilter)])

  def filter_marker_range(self, topmarker, botmarker, topoffset, botoffset ):
    for well in self.cdsdata:
      depthrg = self.wellinfo.depthrange(well)
      topdepth = self.wellinfo.markerdepth(well, topmarker)
      if np.isnan(topdepth):
        topdepth = depthrg[0]
      topdepth -= topoffset
      botdepth = self.wellinfo.markerdepth(well, botmarker)
      if np.isnan(botdepth):
        botdepth = depthrg[-1]
      botdepth += botoffset
      filter = [True if z>=topdepth and z<=botdepth else False for z in self.cdsdata[well].data['depth']]
      self.cdsviews[well].update(filters=[BooleanFilter(filter)])

  
      

      


