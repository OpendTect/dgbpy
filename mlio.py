#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Nov 2018
#
# _________________________________________________________________________
# various tools machine learning data handling
#


import odpy.dbman as oddbman
import dgbpy.keystr as dgbkeys
import dgbpy.hdf5 as dgbhdf5

nladbdirid = '100060'
mltrlgrp = 'Deep Learning Model'
kerastrl = 'Keras'

def getInfo( filenm ):
  return dgbhdf5.getInfo( filenm )

def getTrainingData( filenm, decim=False ):
  ret = { dgbkeys.infodictstr: getInfo( filenm ) }
  examples = dgbhdf5.getAllCubeLets( filenm, decim )
  for ex in examples:
    ret.update({ex: examples[ex]})
  return ret

def getSaveLoc( outnm, args ):
  dblist = oddbman.getDBList(mltrlgrp,args)
  curentry = oddbman.getByName( dblist, outnm )
  if curentry != None:
    return oddbman.getFileLocation(curentry,args)
  return oddbman.getNewEntryFileName(outnm,nladbdirid,mltrlgrp,kerastrl,\
                                     dgbhdf5.hdf5ext,args)
