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

from odpy.common import log_msg, get_log_file
from odpy.oscommand import printProcessTime
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

def computeChunkedScaler_(datasets,infos,groupnm,scalebyattrib):
  chunknb = len(datasets)
  chunkmean = list()
  chunkstd = list()
  chunklen = list()
  for dataset in datasets:
    datasetchunk = dgbmlio.getDatasetsByGroup( dataset, groupnm )
    chunksize = 0
    for keynm in datasetchunk:
      data = datasetchunk[keynm]
      if groupnm in data:
        dsets = data[groupnm]
        for subgroupnm in dsets:
          chunksize += len(dsets[subgroupnm])
    chunklen.append(chunksize)
    scaleronechunk = computeScaler_( datasetchunk, infos, scalebyattrib )
    if scaleronechunk == None:
      chunkmean.append(0)
      chunkstd.append(0)
    else:
      chunkmean.append(scaleronechunk.mean_)
      chunkstd.append(scaleronechunk.scale_)

  if chunknb < 2:
    return getNewScaler( chunkmean[0], chunkstd[0] )
  if scalebyattrib:
    attrnb = dgbhdf5.getNrAttribs(infos)
  else:
    attrnb = 1
  #Calculate Mean and Var
  totalmean = list()
  totalstd = list()
  for attr in range(attrnb):
    attrmeansum = 0
    attrstdsum = 0
    attrsize = 0
    for ichunk in range(chunknb):
      attrchunksize = chunklen[ichunk]
      if attrchunksize > 0:
        attrmeansum += chunkmean[ichunk][attr] * attrchunksize
        attrstdsum += chunkstd[ichunk][attr] * attrchunksize
      attrsize += attrchunksize
    if attrsize > 0:
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
      printProcessTime( 'Scaler computation', True, print_fn=log_msg )
      scaler = computeScaler_( datasets[0], infos, scalebyattrib )
      printProcessTime( 'Scaler computation', False, print_fn=log_msg, withprocline=False )
      for groupnm in inp:
        inp[groupnm].update({dgbkeys.scaledictstr: scaler})
  else:
    for groupnm in inp:
      if dgbmlio.hasScaler( infos, groupnm ) and not force:
        continue
      printProcessTime( 'Scaler computation', True, print_fn=log_msg )
      scaler = computeChunkedScaler_(datasets,infos,groupnm,scalebyattrib)
      printProcessTime( 'Scaler computation', False, print_fn=log_msg, withprocline=False )
      inp[groupnm].update({dgbkeys.scaledictstr: scaler})
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
  if nbchunks > 1: #Decimate, only need to return the updated info
    return {dgbkeys.infodictstr: infos}
  return getScaledTrainingDataByInfo( infos, flatten=flatten, scale=scale )

def getInputList( datasets ):
  ret = {}
  for keynm in datasets:
    dgbhdf5.dictAddIfNew( datasets[keynm], ret )
  return ret.keys()

def getScaledTrainingDataByInfo( infos, flatten=False, scale=True, ichunk=0 ):
  printProcessTime( 'Data pre-loading', True, print_fn=log_msg )
  x_train = list()
  y_train = list()
  x_validate = list()
  y_validate = list()
  datasets = infos[dgbkeys.trainseldicstr][ichunk]
  groups = getInputList( datasets )
  for groupnm in groups:
    dsets = dgbmlio.getDatasetsByGroup( datasets, groupnm )
    ret = dgbmlio.getTrainingDataByInfo( infos, dsets )
    if scale and groupnm in infos[dgbkeys.inputdictstr]:
      scaler = infos[dgbkeys.inputdictstr][groupnm][dgbkeys.scaledictstr]
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
  nrexamples = 0
  if len(x_train)>0:
    x_train = np.concatenate(x_train)
    nrexamples += len(x_train)
    ret.update({dgbkeys.xtraindictstr: x_train })
  if len(y_train)>0:
    ret.update({dgbkeys.ytraindictstr: np.concatenate(y_train) })
  if len(x_validate)>0:
    x_validate = np.concatenate(x_validate)
    nrexamples += len(x_validate)
    ret.update({dgbkeys.xvaliddictstr: x_validate })
  if len(y_validate)>0:
    ret.update({dgbkeys.yvaliddictstr: np.concatenate(y_validate) })

  printProcessTime( 'Data pre-loading', False, print_fn=log_msg, withprocline=False )

  import copy
  decinfos = copy.deepcopy( infos )
  decinfos[dgbkeys.trainseldicstr] = [datasets]
  ret.update({dgbkeys.infodictstr: decinfos})

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
    for iattr in range(nrattribs):
      if nrattribs == 1:
        inp = x_train
      else:
        inp = x_train[:,iattr]
      inp -= scaler.mean_[iattr]
      doscale = np.flatnonzero( scaler.scale_ )
      if (doscale == iattr)[iattr]:
        inp /= scaler.scale_[iattr]


def doTrain( examplefilenm, platform=dgbkeys.kerasplfnm, type=TrainType.New,
             params=None, outnm=dgbkeys.modelnm, logdir=None, clearlogs=False, modelin=None,
             args=None ):
  (model,infos) = (None,None)
  if type == None:
    type = TrainType.New
  if type != TrainType.New:
    (model,infos) = dgbmlio.getModel( modelin, fortrain=True )

  trainingdp = None
  validation_split = 0.2 #Params?
  if platform == dgbkeys.kerasplfnm:
    import dgbpy.dgbkeras as dgbkeras
    import tempfile
    if params == None:
      params = dgbkeras.getParams()
    dgbkeras.set_compute_device( params[dgbkeras.prefercpustr] )
    
    trainingdp = getScaledTrainingData( examplefilenm, flatten=False,
                                        scale=True, force=False,
                                        nbchunks=params['nbchunk'],
                                        split=validation_split )
    logdir = dgbkeras.getLogDir( examplefilenm, logdir, clearlogs, args )
    if type == TrainType.New:
      model = dgbkeras.getDefaultModel(trainingdp[dgbkeys.infodictstr],
                                       type=params['type'],
                                       learnrate=params['learnrate'])
    elif type == TrainType.Transfer:
      model = dgbkeras.transfer( model )

    tempmodelnm = None
    logfnm = get_log_file()
    if logfnm != None:
      tempmodelfnm = tempfile.NamedTemporaryFile( dir=os.path.dirname(logfnm) )
      tempmodelnm = tempmodelfnm.name + '.h5'
      tempmodelfnm = None
    print('--Training Started--', flush=True)
    try:
      model = dgbkeras.train( model, trainingdp, params=params,
                              trainfile=examplefilenm, logdir=logdir,
                              withaugmentation=dgbkeras.withaugmentation,
                              tempnm=tempmodelnm )
    except ResourceExhaustedError:
      model = dgbmlio.getModel( tempmodelnm, True )
    try:
      if os.path.exists(tempmodelnm):
        os.remove( tempmodelnm )
    except:
      pass
  elif platform == dgbkeys.scikitplfnm:
    import dgbpy.dgbscikit as dgbscikit
    if params == None:
      params = dgbscikit.getParams()
    trainingdp = getScaledTrainingData( examplefilenm, flatten=True,
                                        scale=True, force=False,
                                        split=validation_split )
    if type == TrainType.New:
      model = dgbscikit.getDefaultModel( trainingdp[dgbkeys.infodictstr],
                                         params )
    print('--Training Started--', flush=True)
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

def doApply( model, info, samples, scaler=None, applyinfo=None, batchsize=None ):
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
    res = dgbkeras.apply( model, samples, isclassification, withpred, withprobs, withconfidence, doprobabilities, \
                          scaler=None, batch_size=batchsize  )
  elif platform == dgbkeys.scikitplfnm:
    import dgbpy.dgbscikit as dgbscikit
    res = dgbscikit.apply( model, samples, scaler, isclassification, withpred, withprobs, withconfidence, doprobabilities )
  elif platform == dgbkeys.numpyvalstr:
    res = numpyApply( samples )
  else:
    log_msg( 'Unsupported machine learning platform' )
    raise AttributeError

  if isclassification and withpred:
    classarr = res[dgbkeys.preddictstr]
    dgbmlio.unnormalize_class_vector( classarr, info[dgbkeys.classesdictstr] )
    res.update({dgbkeys.preddictstr: classarr})

  return reformat( res, applyinfo )

def numpyApply( samples ):
  import numpy as np
  return {
    dgbkeys.preddictstr: np.mean( samples, axis=(1,2,3,4) )
  }

def inputCount( infos, raw=False, dsets=None ):
  if dsets == None:
    if raw or not dgbkeys.trainseldicstr in infos:
      return inputCount_( infos[dgbkeys.datasetdictstr] )
    else:
      dsets = infos[dgbkeys.trainseldicstr]
  if isinstance(dsets,list):
    return inputCountList(infos,dsets)
  ret = {}
  for keynm in dsets:
    ret.update({keynm: inputCount_(dsets[keynm])})
  return ret

def inputCountList( infos, dsetslist ):
  ret = list()
  for dsets in dsetslist:
    ret.append( inputCount(infos,dsets=dsets) )
  return ret

def inputCount_( dsets ):
  ret = {}
  dscounts = dgbmlio.datasetCount( dsets )
  for groupnm in dscounts:
    if groupnm == 'size':
      continue
    ret.update({groupnm: dscounts[groupnm]['size']})
  return ret

def split( arrays, ratio ):
  if len(arrays) < 1:
    return None
  nrpts = len(arrays[0])
  idxs = np.random.shuffle( np.arange(np.int64(nrpts)) )

