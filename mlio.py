#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : Arnaud
# DATE     : November 2018
#
# various tools machine learning data handling
#


import numpy as np
import random
import dgbpy.hdf5 as dgbhdf5


def getTrainingData( filenm, decim=None ):
  infos = dgbhdf5.getInfo( filenm )
  data = dgbhdf5.getAllCubeLets( filenm, decim )
  return {
    'info': infos,
    'train': data
  }
