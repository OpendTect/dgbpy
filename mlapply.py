#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Jan 2019
#
# _________________________________________________________________________
# various tools for applying machine learning
#

import os

def doTrain( params, outnm, args ):
  trainfile = args['h5file'].name
  import dgbpy.mlio as dgbmlio
  training = dgbmlio.getTrainingData( trainfile, params['decimation'] )
  import dgbpy.dgbkeras as dgbkeras
  model = dgbkeras.getDefaultModel(training['info'])
  model = dgbkeras.train( model, training, params, trainfile=trainfile )
  outfnm = None
  try:
    outfnm = dgbmlio.getSaveLoc( args, outnm )
  except FileNotFoundError:
    raise
  dgbkeras.save( model, trainfile, outfnm )
  return os.path.isfile( outfnm )
