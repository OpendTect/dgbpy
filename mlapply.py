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

modelnm='<new model>'

def doTrain( trainfilenm, platform='keras', params=None, outnm=modelnm, args=None ):
  outfnm = None
  try:
    outfnm = dgbmlio.getSaveLoc( outnm, args )
  except FileNotFoundError:
    raise
  decimate = False
  if params != None and 'decimation' in params:
    decimate = params['decimation']
  training = dgbmlio.getTrainingData( trainfilenm, decimate )
  model = None
  if platform == 'keras':
    import dgbpy.dgbkeras as dgbkeras
    if params == None:
      params = dgbkeras.getParams()
    model = dgbkeras.getDefaultModel(training['info'])
    model = dgbkeras.train( model, training, params, trainfile=trainfilenm )
    dgbkeras.save( model, trainfilenm, outfnm )
  elif platform == 'scikit':
    log_msg( 'scikit platform not supported (yet)' )
    import dgbpy.dgbscikit as dgbscikit
    if params == None:
      params = dgbscikit.getParams()
    raise AttributeError
  else:
    log_msg( 'Unsupported machine learning platform' )
    raise AttributeError
  return (outfnm != None and os.path.isfile( outfnm ))
