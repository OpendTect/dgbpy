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
import dgbpy.keystr as dgbkeys
import dgbpy.hdf5 as dgbhdf5
import dgbpy.mlio as dgbmlio

def doTrain( trainfilenm, platform=dgbkeys.kerasplfnm, params=None, \
             outnm=dgbkeys.modelnm, args=None ):
  outfnm = None
  try:
    outfnm = dgbmlio.getSaveLoc( outnm, args )
  except FileNotFoundError:
    raise
  decimate = False
  if params != None and dgbkeys.decimkeystr in params:
    decimate = params[dgbkeys.decimkeystr]
  training = dgbmlio.getTrainingData( trainfilenm, decimate )
  if platform == dgbkeys.kerasplfnm:
    import dgbpy.dgbkeras as dgbkeras
    if params == None:
      params = dgbkeras.getParams()
    model = dgbkeras.getDefaultModel(training[dgbkeys.infodictstr])
    model = dgbkeras.train( model, training, params, trainfile=trainfilenm )
    dgbkeras.save( model, trainfilenm, outfnm )
  elif platform == dgbkeys.scikitplfnm:
    log_msg( 'scikit platform not supported (yet)' )
    import dgbpy.dgbscikit as dgbscikit
    if params == None:
      params = dgbscikit.getParams()
    raise AttributeError
  else:
    log_msg( 'Unsupported machine learning platform' )
    raise AttributeError
  return (outfnm != None and os.path.isfile( outfnm ))

def doApplyFromFile( modelfnm, samples, outsubsel=None ):
  (model,info) = dgbmlio.getModel( modelfnm )
  applyinfo = dgbmlio.getApplyInfo( info, outsubsel )
  return doApply( model, info, samples, applyinfo )

def doApply( model, info, samples, applyinfo=None ):
  platform = info[dgbkeys.plfdictstr]
  if applyinfo==None:
    applyinfo = dgbmlio.getApplyInfo( info )

  if platform == dgbkeys.kerasplfnm:
    import dgbpy.dgbkeras as dgbkeras
    return dgbkeras.apply( model, samples, applyinfo=applyinfo )
  elif platform == dgbkeys.scikitplfnm:
    log_msg( 'scikit platform not supported (yet)' )
    import dgbpy.dgbscikit as dgbscikit
    raise AttributeError
  elif platform == dgbkeys.numpyvalstr:
    return numpyApply( samples )
  else:
    log_msg( 'Unsupported machine learning platform' )
    raise AttributeError

def numpyApply( samples ):
  import numpy as np
  return {
    dgbkeys.preddictstr: np.mean( samples, axis=(1,2,3,4) )
  }

