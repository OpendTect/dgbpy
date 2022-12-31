import logging
from bokeh.io import curdoc

import dgbpy.keystr as dbk
from dgbpy import uisklearn, uitorch, uikeras

class MsgHandler(logging.StreamHandler):
  def __init__(self):
    logging.StreamHandler.__init__(self)
    self.msginfo = {}
    self.servmgr = None

  def add_servmgr(self, servmgr):
    self.servmgr = servmgr

  def add(self, msgstr, msgkey, msgjson):
    self.msginfo[msgstr] = {'msgkey': msgkey, 'jsonobj': msgjson}

  def add_callback(self, callbacks):
    for cb in callbacks:
      self.msginfo[cb] = {'callback': callbacks[cb]}

  def emit(self, record):
    try:
      logmsg = self.format(record)
      for msgstr, value in self.msginfo.items():
        if msgstr in logmsg:
          if 'callback' in value:
            self.docallback(logmsg, msgstr)
          else:
            self.sendmsg(msgstr)
    except (KeyboardInterrupt, SystemExit):
      raise
    except:
      self.handleError(record)

  def sendmsg(self, msgnm):
    if self.servmgr:
      self.servmgr.sendObject(self.msginfo[msgnm]['msgkey'], self.msginfo[msgnm]['jsonobj'])

  def docallback(self, logmsg, msgstr):
    self.msginfo[msgstr]['callback'](logmsg)

def get_default_examples():
  retinfo = {
    'Dummy': {
                'target': 'Dummy',
                'id': 0,
                'collection': {'Dummy': {'dbkey': '100050.1', 'id': 0}
              }
    }
  }
  return retinfo

def get_default_input():
  retinfo = {
    'Dummy': {
                'collection': {'Dummy': {'id': 0}},
                'id': 0,
                'scale': uisklearn.StandardScaler()
              }
  }
  return retinfo

def get_default_info():
  retinfo = {
    dbk.learntypedictstr: dbk.loglogtypestr,
    dbk.segmentdictstr: False,
    dbk.inpshapedictstr: 1,
    dbk.outshapedictstr: 1,
    dbk.classdictstr: False,
    dbk.interpoldictstr: False,
    dbk.exampledictstr: get_default_examples(),
    dbk.inputdictstr: get_default_input(),
    dbk.filedictstr: 'dummy',
    dbk.estimatedsizedictstr: 1
  }
  return retinfo

def get_platforms():
  mlplatform = []
  if uikeras.hasKeras():
    mlplatform.append( uikeras.getPlatformNm(True) )
  if uitorch.hasTorch():
    mlplatform.append( uitorch.getPlatformNm(True) )
  if uisklearn.hasScikit():
    mlplatform.append( uisklearn.getPlatformNm(True) )
  if not bool(mlplatform):
    mlplatform.append(uinoplfm().getPlatformNm(True) )
  return mlplatform

def get_default_platform(mllearntype=dbk.loglogtypestr):
    if mllearntype == dbk.loglogtypestr or \
      mllearntype == dbk.logclustertypestr or \
      mllearntype == dbk.seisproptypestr:
      if uisklearn.hasScikit():
        return uisklearn.getPlatformNm(True)[0]
    else:
      if uikeras.hasKeras():
        return uikeras.getPlatformNm(True)[0]
      if uitorch.hasTorch():
        return uitorch.getPlatformNm(True)[0]
    return uinoplfm().getPlatformNm(True)[0]

class uinoplfm:
  def __init__(self):
    self.platform = ('No platform', 'No Available Platform')

  def getPlatformNm(self, full=False ):
    if full:
      return self.platform
    return self.platform[0]