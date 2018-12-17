#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : Arnaud
# DATE     : November 2018
#
# various tools machine learning data handling
#


import sys
from odpy.common import *
import odpy.dbman as oddbman
import dgbpy.hdf5 as dgbhdf5

nladbdirid = '100060'
mltrlgrp = 'Deep Learning Model'
kerastrl = 'Keras'
hdf5ext = 'h5'

def getTrainingData( filenm, decim=None ):
  infos = dgbhdf5.getInfo( filenm )
  data = dgbhdf5.getAllCubeLets( filenm, decim )
  return {
    'info': infos,
    'train': data
  }


def getSaveLoc( args, outnm ):
  dblist = oddbman.getDBList(args,mltrlgrp)
  curentry = oddbman.getByName( dblist, outnm )
  if curentry != None:
    return oddbman.getFileLocation(args,curentry)
  return oddbman.getNewEntryFileName(args,outnm,nladbdirid,mltrlgrp,kerastrl,hdf5ext)
