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

import config as cfg


class WellInfo:
  COLS = ["name","x", "y", "logs", "markers", "mdmin", "mdmax"]

  def __init__(self):
    self.cds = ColumnDataSource(data={cl: [] for cl in WellInfo.COLS})
    self.get_data()

  def get_data(self):
    wellnms = odwm.getNames(reload=True)
    xloc = []
    yloc = []
    logs = []
    markers = []
    mdmin = []
    mdmax = []
    for wellnm in wellnms:
      welldata = odwm.getInfo(wellnm)
      xloc.append(welldata["X"])
      yloc.append(welldata["Y"])
      logs.append(odwm.getLogNames(wellnm))
      markers.append(odwm.getMarkers(wellnm))
      mdrng = odwm.getTrack(wellnm)[0]
      mdmin.append(mdrng[0])
      mdmax.append(mdrng[-1])
    self.cds.update(data={'name': wellnms, 'x': xloc, 'y': yloc, 'logs': logs, 'markers': markers, 'mdmin': mdmin, 'mdmax': mdmax})

  def names(self):
    return self.cds.data['name']

  def lognames(self, well):
    try:
      return self.cds.data['logs'][self.names().index(well)]
    except ValueError:
      return []

  def markernames(self, well):
    try:
      return self.cds.data['markers'][self.names().index(well)][0]
    except ValueError:
      return []

  def markerdepths(self, well):
    try:
      return self.cds.data['markers'][self.names().index(well)][1]
    except ValueError:
      return []

  def markerdepth(self, well, marker):
    try:
      idx = self.markernames(well).index(marker)
      return self.markerdepths(well)[idx]
    except ValueError:
      return np.nan

  def depthrange(self, well):
    depthrg = [0,1000]
    idx = None
    try:
      idx = self.names().index(well)
      depthrg = [self.cds.data['mdmin'][idx], self.cds.data['mdmax'][idx]]
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
    if well in self.names():
      idxs = self.logidxs(well, logsel)
      idxstr = ','.join(str(x) for x in idxs)
      if idxstr:
        data = odwm.getLogs(well, idxstr)
        logkys = []
        for ky in data:
          vals = np.array(data[ky],dtype=float)
          vals[vals==1e30] = np.nan
          data[ky] = vals
          logkys.append(ky)
        data['logkeys'] = logkys
      else:
        mdrng = self.depthrange(well)
        data['depth'] = np.arange(mdrng[0], mdrng[-1], 1)
      nv = data['depth'].size
      data['well'] = np.ones(nv)*self.names().index(well)
    return data

  def get_common_markernames(self, wellnms):
    mrkset = []
    for well in wellnms:
      mrknms = self.markernames(well)
      if mrkset:
        mrkset = [nm for nm in mrkset if nm in mrknms]
      else:
        mrkset = mrknms
    return mrkset

  def get_unique_markernames(self, wellnms):
    mrkset = set()
    for well in wellnms:
      mrknms = self.markernames(well)
      mrkset.update(mrknms)
    return list(mrkset)

class WellCrossplotData:
  COLS = ["well", "depth", "xlog", "ylog"]
  def __init__(self):
    self.logsel = []
    self.wellinfo = WellInfo()
    self.wellnms = []
    self.cds = ColumnDataSource(data={cl: [] for cl in WellCrossplotData.COLS})
    self.cdsview = CDSView(filters=[], source=self.cds )

  def get_data(self, wellnms, xlog, ylog):
    new_data = {}
    self.wellnms = wellnms
    for well in wellnms:
      data = self.wellinfo.get_logdata(well, [xlog, ylog])
      if not data:
        return
      logkys = data.get('logkeys', [])
      xlogky = 'depth'
      if xlog!='depth' and logkys:
        xlogky = logkys[1]
      ylogky = 'depth'
      if ylog!='depth' and logkys:
        ylogky = logkys[-1]
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
        self.logsel = [xlogky, ylogky]
    self.cds.update(data=new_data)

  def filter_reset(self):
    self.cdsview.update(filters=[])

  def filter_depth_range(self, mindepth, maxdepth):
    zfilter = [True if z>=mindepth and z<=maxdepth else False for z in self.cds.data['depth']]
    self.cdsview.update(filters=[BooleanFilter(zfilter)])

  def filter_marker_range(self, topmarker, botmarker, topoffset, botoffset ):
    filter = None
    data = self.cds.data
    allwells = self.wellinfo.names()
    for well in self.wellnms:
      topdepth = self.wellinfo.markerdepth(well, topmarker)
      if np.isnan(topdepth):
        topdepth = data['depth'][0]
      topdepth -= topoffset
      botdepth = self.wellinfo.markerdepth(well, botmarker)
      if np.isnan(botdepth):
        botdepth = data['depth'][-1]
      botdepth += botoffset
      wellidx = allwells.index(well)
      wellfilter = [True if z>=topdepth and z<=botdepth and widx==wellidx else False for z, widx in zip(data['depth'],data['well'])]
      if filter is None:
        filter = wellfilter
      else:
        filter += wellfilter
    self.cdsview.update(filters=[BooleanFilter(filter)])

  
      

      


