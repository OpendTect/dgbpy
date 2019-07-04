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

def doTrain( examplefilenm, platform=dgbkeys.kerasplfnm, params=None, \
             outnm=dgbkeys.modelnm, args=None ):
  outfnm = dgbmlio.getSaveLoc( outnm, args )
  decimate = False
  if params != None and dgbkeys.decimkeystr in params:
    decimate = params[dgbkeys.decimkeystr]
  training = dgbmlio.getTrainingData( examplefilenm, decimate )
  if platform == dgbkeys.kerasplfnm:
    import dgbpy.dgbkeras as dgbkeras
    if params == None:
      params = dgbkeras.getParams()
    model = dgbkeras.getDefaultModel(training[dgbkeys.infodictstr])
    model = dgbkeras.train( model, training, params, trainfile=examplefilenm )
    dgbkeras.save( model, examplefilenm, outfnm )
  elif platform == dgbkeys.scikitplfnm:
    log_msg( 'scikit platform not supported (yet)' )
    import dgbpy.dgbscikit as dgbscikit
    if params == None:
      params = dgbscikit.getParams()
    #(model,scaler) = dgbscikit.train( training, params, trainfile=examplefilenm )
    #dgbscikit.save( model, examplefilenm, outfnm, scaler=scaler )
    raise AttributeError
  else:
    log_msg( 'Unsupported machine learning platform' )
    raise AttributeError
  return (outfnm != None and os.path.isfile( outfnm ))

def reformat( res, applyinfo ):
  if dgbkeys.preddictstr in res:
    res[dgbkeys.preddictstr] = res[dgbkeys.preddictstr].astype( applyinfo[dgbkeys.dtypepred] )
  if dgbkeys.probadictstr in res:
    res[dgbkeys.probadictstr] = res[dgbkeys.probadictstr].astype( applyinfo[dgbkeys.dtypeprob] )
  if dgbkeys.confdictstr in res:
    res[dgbkeys.confdictstr] = res[dgbkeys.confdictstr].astype( applyinfo[dgbkeys.dtypeconf] )
  return res

def doApplyFromFile( modelfnm, samples, outsubsel=None ):
  (model,info) = dgbmlio.getModel( modelfnm )
  applyinfo = dgbmlio.getApplyInfo( info, outsubsel )
  return doApply( model, info, samples, applyinfo )

def doApply( model, info, samples, scaler=None, applyinfo=None ):
  platform = info[dgbkeys.plfdictstr]
  if applyinfo == None:
    applyinfo = dgbmlio.getApplyInfo( info )

  isclassification = info[dgbkeys.classdictstr]
  withpred = dgbkeys.dtypepred in applyinfo
  withprobs = False
  doprobabilities = False
  withconfidence = False
  if isclassification:
    if dgbkeys.probadictstr in applyinfo:
      withprobs = applyinfo[dgbkeys.probadictstr]
      doprobabilities = len(withprobs) > 0
    withconfidence = dgbkeys.dtypeconf in applyinfo

  res = None
  if platform == dgbkeys.kerasplfnm:
    import dgbpy.dgbkeras as dgbkeras
    res = dgbkeras.apply( model, samples, isclassification, withpred, withprobs, withconfidence, doprobabilities )
  elif platform == dgbkeys.scikitplfnm:
    import dgbpy.dgbscikit as dgbscikit
    res = dgbscikit.apply( model, samples, scaler, isclassification, withpred, withprobs, withconfidence, doprobabilities )
  elif platform == dgbkeys.numpyvalstr:
    res = numpyApply( samples )
  else:
    log_msg( 'Unsupported machine learning platform' )
    raise AttributeError

  return reformat( res, applyinfo )

def numpyApply( samples ):
  import numpy as np
  return {
    dgbkeys.preddictstr: np.mean( samples, axis=(1,2,3,4) )
  }

