#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : Arnaud
# DATE     : November 2018
#
# various tools machine learning data handling
#


import sys

import odpy.dbman as oddbman
import dgbpy.hdf5 as dgbhdf5

nladbdirid = '100060'
mltrlgrp = 'Deep Learning Model'
kerastrl = 'Keras'
hdf5ext = 'h5'

def getTrainingData( filenm, decim=False ):
  infos = dgbhdf5.getInfo( filenm )
  data = dgbhdf5.getAllCubeLets( filenm, decim )
  return {
    'info': infos,
    'train': data
  }

def getSaveLoc( outnm, args ):
  dblist = oddbman.getDBList(mltrlgrp,args)
  curentry = oddbman.getByName( dblist, outnm )
  if curentry != None:
    return oddbman.getFileLocation(curentry,args)
  return oddbman.getNewEntryFileName(outnm,nladbdirid,mltrlgrp,kerastrl,hdf5ext,args)
