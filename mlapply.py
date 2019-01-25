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

def doApply( modelfnm, samples, outputs=None, platform=None, type=None,
             isclassification=None ):
  if platform == None or type == None or isclassification == None:
    import dgbpy.hdf5 as dgbhdf5
    infos = dgbhdf5.getInfo( modelfnm )
    if platform == None:
      platform = infos[dgbhdf5.plfdictstr]
    if type == None:
      type = infos[dgbhdf5.typedictstr]
    if isclassification == None:
      isclassification = infos[dgbhdf5.classdictstr]
  withclass = isclassification and \
              (outputs==None or dgbhdf5.classvalstr in outputs)
  withconfidence = isclassification and \
                   (outputs==None or dgbhdf5.confvalstr in outputs)
  if isclassification:
    withprobs = dgbhdf5.getClassIndices( modelfnm, outputs )

  if platform == dgbkeys.kerasplfnm:
    import dgbpy.dgbkeras as dgbkeras
    model = dgbkeras.load( modelfnm )
    return dgbkeras.apply( model, samples, isclassification,
                           withclass=withclass, withprobs=withprobs,
                           withconfidence=withconfidence )
  elif platform == dgbkeys.scikitplfnm:
    log_msg( 'scikit platform not supported (yet)' )
    import dgbpy.dgbscikit as dgbscikit
    raise AttributeError
  else:
    log_msg( 'Unsupported machine learning platform' )
    raise AttributeError

