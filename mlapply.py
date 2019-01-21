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

from odpy.common import log_msg
import dgbpy.mlio as dgbmlio

def doTrain( platform, params, outnm, args ):
  trainfile = args['h5file'].name
  training = dgbmlio.getTrainingData( trainfile, params['decimation'] )
  outfnm = None
  if platform == 'keras':
    import dgbpy.dgbkeras as dgbkeras
    model = dgbkeras.getDefaultModel(training['info'])
    model = dgbkeras.train( model, training, params, trainfile=trainfile )
    try:
      outfnm = dgbmlio.getSaveLoc( args, outnm )
    except FileNotFoundError:
      raise
    dgbkeras.save( model, trainfile, outfnm )
  elif platform == 'scikit':
    log_msg( 'scikit platform not supported (yet)' )
    raise AttributeError
  else:
    log_msg( 'Unsupported machine learning platform' )
    raise AttributeError
  return (outfnm != None and os.path.isfile( outfnm ))
