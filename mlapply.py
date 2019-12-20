#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Jan 2019
#
# _________________________________________________________________________
# various tools for applying machine learning
#

from enum import Enum
import numpy as np
import os

from odpy.common import log_msg
import dgbpy.keystr as dgbkeys
import dgbpy.hdf5 as dgbhdf5
import dgbpy.mlio as dgbmlio

TrainType = Enum( 'TrainType', 'New Resume Transfer', module=__name__ )

def computeScaler_( datasets, infos, scalebyattrib ):
  ret = dgbmlio.getTrainingDataByInfo( infos, datasets )
  allx = list()
  if dgbkeys.xtraindictstr in ret:
    if len(ret[dgbkeys.xtraindictstr]) > 0:
      allx.append( ret[dgbkeys.xtraindictstr] )
  if dgbkeys.xvaliddictstr in ret:
    if len(ret[dgbkeys.xvaliddictstr]) > 0:
      allx.append( ret[dgbkeys.xvaliddictstr] )
  if len(allx) > 0:
    x_data = np.concatenate( allx )
    return getScaler( x_data, byattrib=scalebyattrib )
  return None

def computeChunkedScaler_(datasets,infos,inputnm,scalebyattrib):
  chunknb = len(datasets)
  chunkmean = list()
  chunkstd = list()
  chunklen = list()
  for dataset in datasets:
    datasetchunk = dgbmlio.getDatasetsByInput( dataset, inputnm )
    chunksize = 0
    for dsetnm in datasetchunk:
      data = datasetchunk[dsetnm]
      for groupnm in data:
        inp = data[groupnm]
        chunksize += len(inp[inputnm])
    chunklen.append(chunksize)
    scaleronechunk = computeScaler_( datasetchunk, infos, scalebyattrib )
    chunkmean.append(scaleronechunk.mean_)
    chunkstd.append(scaleronechunk.scale_)
  if chunknb < 2:
    return getNewScaler( chunkmean[0], chunkstd[0] )
  attrnb = len(chunkmean[0])
  #Calculate Mean and Var
  totalmean = list()
  totalstd = list()
  for attr in range(attrnb):
    attrmeansum = 0
    attrstdsum = 0
    attrsize = 0
    for ichunk in range(chunknb):
      attrchunksize = chunklen[ichunk]
      attrmeansum += chunkmean[ichunk][attr] * attrchunksize
      attrstdsum += chunkstd[ichunk][attr] * attrchunksize
      attrsize += attrchunksize
    attrmeansum /= attrsize
    attrstdsum /= attrsize
    totalmean.append( attrmeansum )
    totalstd.append( attrstdsum )
  return getNewScaler( totalmean, totalstd )

def computeScaler( infos, scalebyattrib, force=False ):
  datasets = infos[dgbkeys.trainseldicstr]
  inp = infos[dgbkeys.inputdictstr]
  if infos[dgbkeys.learntypedictstr] == dgbkeys.loglogtypestr:
    if not dgbmlio.hasScaler(infos) or force:
      scaler = computeScaler_( datasets[0], infos, scalebyattrib )
      for inputnm in inp:
        inp[inputnm].update({dgbkeys.scaledictstr: scaler})
  else:
    for inputnm in inp:
      if dgbmlio.hasScaler( infos, inputnm ) and not force:
        continue
      scaler = computeChunkedScaler_(datasets,infos,inputnm,scalebyattrib)
      inp[inputnm].update({dgbkeys.scaledictstr: scaler})
  return infos

def getScaledTrainingData( filenm, flatten=False, scale=True, force=False, 
                           nbchunks=1, split=None ):
  if isinstance(scale,bool):
    doscale = scale
    scalebyattrib = True
  else:
    doscale = scale[0]
    scalebyattrib = len(scale) < 2 or scale[1]

  infos = dgbmlio.getInfo( filenm )
  dsets = dgbmlio.getChunks(infos[dgbkeys.datasetdictstr],nbchunks)
  datasets = []
  for dset in dsets:
    datasets.append( dgbmlio.getDatasetNms(dset,validation_split=split) )
  infos.update({dgbkeys.trainseldicstr: datasets})
  if doscale:
    infos = computeScaler( infos, scalebyattrib, force )
  decimate = nbchunks > 1
  ret = {}
  ret.update({dgbkeys.infodictstr: infos})
  if decimate: #Decimate, only need to update output information
    y_examples = list()
    for dataset in datasets:
      examples = dgbhdf5.getDatasets( infos, dataset )
      if dgbkeys.ytraindictstr in examples:
        y_examples.append( examples[dgbkeys.ytraindictstr] )
      if dgbkeys.yvaliddictstr in examples:
        y_examples.append( examples[dgbkeys.yvaliddictstr] )
    ret.update({ dgbkeys.infodictstr: dgbmlio.getClasses(infos,y_examples) })
    return ret
  return getScaledTrainingDataByInfo( infos, flatten=flatten, scale=scale )

def getScaledTrainingDataByInfo( infos, flatten=False, scale=True, ichunk=0 ):
  x_train = list()
  y_train = list()
  x_validate = list()
  y_validate = list()
  datasets = infos[dgbkeys.trainseldicstr][ichunk]
  inp = infos[dgbkeys.inputdictstr]
  for inputnm in inp:
    input = inp[inputnm]
    dsets = dgbmlio.getDatasetsByInput( datasets, inputnm )
    ret = dgbmlio.getTrainingDataByInfo( infos, dsets )
    if scale:
      scaler = infos[dgbkeys.inputdictstr][inputnm][dgbkeys.scaledictstr]
      if dgbkeys.xtraindictstr in ret:
        transform( ret[dgbkeys.xtraindictstr], scaler )
      if dgbkeys.xvaliddictstr in ret:
        transform( ret[dgbkeys.xvaliddictstr], scaler )
    if dgbkeys.xtraindictstr in ret:
      if len(ret[dgbkeys.xtraindictstr]) > 0:
        x_train.append( ret[dgbkeys.xtraindictstr] )
    if dgbkeys.xvaliddictstr in ret:
      if len(ret[dgbkeys.xvaliddictstr]) > 0:
        x_validate.append( ret[dgbkeys.xvaliddictstr] )
    if dgbkeys.ytraindictstr in ret:
      if len(ret[dgbkeys.ytraindictstr]) > 0:
        y_train.append( ret[dgbkeys.ytraindictstr] )
    if dgbkeys.yvaliddictstr in ret:
      if len(ret[dgbkeys.yvaliddictstr]) > 0:
        y_validate.append( ret[dgbkeys.yvaliddictstr] )
  if len(x_train)>0:
    ret.update({dgbkeys.xtraindictstr: np.concatenate(x_train) })
  if len(y_train)>0:
    ret.update({dgbkeys.ytraindictstr: np.concatenate(y_train) })
  if len(x_validate)>0:
    ret.update({dgbkeys.xvaliddictstr: np.concatenate(x_validate) })
  if len(y_validate)>0:
    ret.update({dgbkeys.yvaliddictstr: np.concatenate(y_validate) })

  if not flatten:
    return ret

  if dgbkeys.xtraindictstr in ret:
    x_train = ret[dgbkeys.xtraindictstr]
    ret[dgbkeys.xtraindictstr] = np.reshape( x_train, (len(x_train),-1) )
  if dgbkeys.xvaliddictstr in ret:
    x_validate = ret[dgbkeys.xvaliddictstr]
    ret[dgbkeys.xvaliddictstr] = np.reshape( x_validate, (len(x_validate),-1) )
  return ret

def getScaler( x_train, byattrib=True ):
  import dgbpy.dgbscikit as dgbscikit
  return dgbscikit.getScaler( x_train, byattrib )

def getNewScaler( mean, scale ):
  import dgbpy.dgbscikit as dgbscikit
  return dgbscikit.getNewScaler( mean, scale )

def transform(x_train,scaler):
  nrattribs = scaler.n_samples_seen_
  if nrattribs > 0:
    for a in range(nrattribs):
      if nrattribs == 1:
        inp = x_train
      else:
        inp = x_train[:,a]
      inp -= scaler.mean_[a]
      doscale = np.flatnonzero( scaler.scale_ )
      if (doscale == a)[a]:
        inp /= scaler.scale_[a]


def doTrain( examplefilenm, platform=dgbkeys.kerasplfnm, type=TrainType.New,
             params=None, outnm=dgbkeys.modelnm, logdir=None, modelin=None,
             args=None ):
  (model,infos) = (None,None)
  if type == None:
    type = TrainType.New
  if type != TrainType.New:
    (model,infos) = dgbmlio.getModel( modelin, fortrain=True )

  trainingdp = None
  if platform == dgbkeys.kerasplfnm:
    import dgbpy.dgbkeras as dgbkeras
    if params == None:
      params = dgbkeras.getParams()
    validation_split = 0.2 #Params?
    trainingdp = getScaledTrainingData( examplefilenm, flatten=False,
                                        scale=True, force=False,
                                        nbchunks=params['nbchunk'],
                                        split=validation_split )
    logdir = dgbkeras.getLogDir( logdir, args )
    if type == TrainType.New:
      model = dgbkeras.getDefaultModel(trainingdp[dgbkeys.infodictstr],
                                       type=params['type'],
                                       learnrate=params['learnrate'])
    elif type == TrainType.Transfer:
      model = dgbkeras.transfer( model )
    model = dgbkeras.train( model, trainingdp, params,
                            trainfile=examplefilenm, logdir=logdir )
  elif platform == dgbkeys.scikitplfnm:
    import dgbpy.dgbscikit as dgbscikit
    if params == None:
      params = dgbscikit.getParams()
    trainingdp = getScaledTrainingData( examplefilenm, flatten=True,
                                        scale=True, force=False )
    if type == TrainType.New:
      model = dgbscikit.getDefaultModel( trainingdp[dgbkeys.infodictstr],
                                         params )
    model = dgbscikit.train( model, trainingdp )
  else:
    log_msg( 'Unsupported machine learning platform' )
    raise AttributeError

  infos = trainingdp[dgbkeys.infodictstr]
  modtype = dgbmlio.getModelType( infos )
  outfnm = dgbmlio.getSaveLoc( outnm, modtype, args )
  dgbmlio.saveModel( model, examplefilenm, platform, infos, outfnm )
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
  (model,info) = dgbmlio.getModel( modelfnm, fortrain=False )
  applyinfo = dgbmlio.getApplyInfo( info, outsubsel )
  return doApply( model, info, samples, applyinfo=applyinfo )

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

def inputCount( dsets, infos ):
  inputnms = infos[dgbkeys.inputdictstr]
  ret = {}
  for ex in dsets:
    ret.update({ex: inputCount_({'key': dsets[ex]}, inputnms)})
  return ret

def inputCount_( dsets, inputnms ):
  ret = {}
  for inputnm in inputnms:
    alldsets = dgbmlio.getDatasetsByInput( dsets, inputnm )
    nbbyinp = 0
    for ex in alldsets:
      dset = alldsets[ex]
      for groupnm in dset:
        for inp in dset[groupnm]:
          nbbyinp += len(dset[groupnm][inp])
    if nbbyinp>0:
      ret.update({inputnm: nbbyinp})
  return ret

def split( arrays, ratio ):
  if len(arrays) < 1:
    return None
  nrpts = len(arrays[0])
  idxs = np.random.shuffle( np.arange(np.int64(nrpts)) )

