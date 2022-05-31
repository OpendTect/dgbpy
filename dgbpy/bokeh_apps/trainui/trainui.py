import logging
from sklearn.preprocessing import StandardScaler
from bokeh.io import curdoc

import dgbpy.keystr as dbk
from dgbpy import uisklearn, uitorch, uikeras

class MsgHandler(logging.StreamHandler):
  def __init__(self, msgstr, servmgr, msgkey, msgjson):
    logging.StreamHandler.__init__(self)
    self.msginfo = {}
    self.add(msgstr, msgkey, msgjson)
    self.servmgr = servmgr

  def add(self, msgstr, msgkey, msgjson):
    self.msginfo[msgstr] = {'msgkey': msgkey, 'jsonobj': msgjson}

  def emit(self, record):
    try:
      logmsg = self.format(record)
      for msgstr in self.msginfo.keys():
        if msgstr in logmsg:
          curdoc().add_next_tick_callback(partial(self.sendmsg, msgnm=msgstr))
    except (KeyboardInterrupt, SystemExit):
      raise
    except:
      self.handleError(record)

  def sendmsg(self, msgnm):
    self.servmgr.sendObject(self.msginfo[msgnm]['msgkey'], self.msginfo[msgnm]['jsonobj'])


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
                'scale': StandardScaler()
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
  mlplatform.append( uikeras.getPlatformNm(True) )
  mlplatform.append( uitorch.getPlatformNm(True) )
  mlplatform.append( uisklearn.getPlatformNm(True) )
  return mlplatform