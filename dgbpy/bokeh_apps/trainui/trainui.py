import logging

import dgbpy.keystr as dbk
from dgbpy import uisklearn


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
    dbk.estimatedsizedictstr: 1,
    dbk.inpscalingdictstr : dbk.globalstdtypestr
  }
  return retinfo
