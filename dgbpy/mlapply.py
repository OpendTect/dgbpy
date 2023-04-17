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
  """ Computes scaler

  Parameters:
    * datasets (dict): dataset
    * infos (dict): information about example file
    * scalebyattrib (bool):
  """

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

def getScaledTrainingData( filenm, flatten=False, scaler=dgbkeys.globalstdtypestr, force=False, 
                           nbchunks=1, split=1, nbfolds=5, seed=None ):
  """ Gets scaled training data

  Parameters:
    * filenm (str): path to file
    * flatten (bool):
    * scale (bool or iter):
    * nbchunks (int): number of data chunks to be created
    * split (float): size of validation data (between 0-1)
  """

  infos = dgbmlio.getInfo( filenm )
  dsets = dgbmlio.getChunks(infos[dgbkeys.datasetdictstr],nbchunks)
  datasets = []
  for dset in dsets:
    if dgbhdf5.isLogInput(infos) and nbfolds:
      datasets.append( dgbmlio.getCrossValidationIndices(dset,seed=seed,valid_inputs=split,nbfolds=nbfolds) )
    else:
      datasets.append( dgbmlio.getDatasetNms(dset, validation_split=split) )
  infos.update({dgbkeys.trainseldicstr: datasets, dgbkeys.seeddictstr: seed})

  scaler, doscale = dgbhdf5.isDefaultScaler(scaler, infos)
  scalebyattrib = doscale
  infos = dgbhdf5.updateScaleInfo(scaler, infos)
  if doscale:
    infos = computeScaler( infos, scalebyattrib, force )
  #Decimate and cross validation, only need to return the updated info
  if nbchunks > 1 or dgbhdf5.isCrossValidation(infos): 
    return {dgbkeys.infodictstr: infos}
  return getScaledTrainingDataByInfo( infos, flatten=flatten, scale=doscale )

def getInputList( datasets ):
  """

  Parameters:
    * datasets (dict): dataset from example file

  Returns:
    * dict:
  """

  ret = {}
  for keynm in datasets:
    dgbhdf5.dictAddIfNew( datasets[keynm], ret )
  return ret.keys()

def getScaledTrainingDataByInfo( infos, flatten=False, scale=True, ichunk=0, ifold=None ):
  """ Gets scaled training data

  Parameters:
    * infos (dict): information about example file
    * flatten (bool):
    * scale (bool): defaults to True, a scaling object is applied to returned data, otherwise if False is specified
    * ichunk (int): number of data chunks to be created

  Returns:
    * dict: of training data with x_train, y_train, x_validation, y_validation, infos as keys.
  """

  printProcessTime( 'Data pre-loading', True, print_fn=log_msg )
  x_train = list()
  y_train = list()
  x_validate = list()
  y_validate = list()
  if ifold and dgbhdf5.isCrossValidation( infos ):
    datasets = infos[dgbkeys.trainseldicstr][ichunk][dgbkeys.foldstr+f'{ifold}']
  else:
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
  """ Gets scaler object for data scaling

  Parameters:
    * x_train (array): data to be scaled
    * byattrib (bool): True if scaling should be done by individual attribute
                       present in data, False if otherwise

  Returns:
    * object: StandardScaler object fitted on data (from sklearn.preprocessing)
  """

  import dgbpy.dgbscikit as dgbscikit
  return dgbscikit.getScaler( x_train, byattrib )

def getNewScaler( mean, scale ):
  """ Gets new scaler object

  Parameters:
    * mean (ndarray of shape (n_features,) or None): mean value to be used for scaling
    * scale ndarray of shape (n_features,) or None: Per feature relative scaling of the
      data to achieve zero mean and unit variance (fromm sklearn docs)

  Returns:
    * object: scaler (an instance of sklearn.preprocessing..StandardScaler())
  """

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
             bokeh=None, args=None):
  """ Method to perform a training job using any platform and for any workflow
      (trained model is also saved)

  Parameters:
    * examplefilenm (str): file name/path to example file in hdf5 format
    * platform (str): machine learning platform choice (options are; keras, scikit-learn, torch)
    * type (str): type of training; new or transfer, or continue (Resume)
    * params (dict): machine learning hyperparameters or parameters options
    * outnm (str): name to save trained model as
    * logdir (str): the path of the directory where to save the log
                    files to be parsed by TensorBoard (only applicable
                    for the keras platform)
    * clearlogs (bool): clears previous logs if any when set to True
    * modelin (str): model file path/name in hdf5 format
    * args (dict, optional):
      Dictionary with the members 'dtectdata' and 'survey' as
      single element lists, and/or 'dtectexec' (see odpy.common.getODSoftwareDir)

  Returns:
    *

  """
  try:
    (model,infos) = (None,None)
    if type == None:
      type = TrainType.New
    if type != TrainType.New:
      (model,infos) = dgbmlio.getModel( modelin, fortrain=True, pars=params )

    trainingdp = None
    validation_split = 0.2 #Params?
    if platform == dgbkeys.kerasplfnm:
      import dgbpy.dgbkeras as dgbkeras
      import tempfile
      if params == None:
        params = dgbkeras.getParams()
      dgbkeras.set_compute_device( params[dgbkeys.prefercpustr] )
      if params['tofp16']:
        dgbkeras.use_mixed_precision()

      trainingdp = getScaledTrainingData( examplefilenm, flatten=False,
                                          scaler=params[dgbkeys.scaledictstr],
                                          force=False,
                                          nbchunks=params['nbchunk'],
                                          split=params['split'],nbfolds=params['nbfold'] )
      tblogdir=None
      if 'withtensorboard' in params and params['withtensorboard']:
        tblogdir = dgbhdf5.getLogDir(dgbkeras.withtensorboard, examplefilenm, platform, logdir, clearlogs, args )
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
      cbfn = None
      if bokeh: cbfn = dgbkeras.BokehProgressCallback
      try:
        model = dgbkeras.train( model, trainingdp, params=params,
                                trainfile=examplefilenm, silent=True, cbfn = cbfn, logdir=tblogdir,tempnm=tempmodelnm )
      except (TypeError,MemoryError) as e:
        if tempmodelnm != None and os.path.exists(tempmodelnm):
          model = dgbmlio.getModel( tempmodelnm, True )
          raise e
      try:
        if os.path.exists(tempmodelnm):
          os.remove( tempmodelnm )
      except:
        pass

    elif platform == dgbkeys.torchplfnm:
      import dgbpy.dgbtorch as dgbtorch
      if params == None:
        params = dgbtorch.getParams()
      tblogdir = None
      if 'withtensorboard' in params and params['withtensorboard']:
        tblogdir = dgbhdf5.getLogDir(dgbtorch.withtensorboard, examplefilenm, platform, logdir, clearlogs, args )
      trainingdp = getScaledTrainingData( examplefilenm, flatten=False,
                                          scaler=params[dgbkeys.scaledictstr],
                                          nbchunks=params['nbchunk'],
                                          force=False,
                                          split=params['split'],nbfolds=params['nbfold'] )

      if type == TrainType.New:
        model = dgbtorch.getDefaultModel(trainingdp[dgbkeys.infodictstr], type=params['type']
                                        )
      elif type == TrainType.Transfer:
        model = dgbtorch.transfer( model )

      print('--Training Started--', flush=True)
      cbfn = None
      if bokeh: cbfn = [dgbtorch.tc.BokehProgressCallback()]
      model = dgbtorch.train(model=model, imgdp=trainingdp, cbfn=cbfn, params=params, logdir=tblogdir, silent=bokeh)

    elif platform == dgbkeys.scikitplfnm:
      import dgbpy.dgbscikit as dgbscikit
      if params == None:
        params = dgbscikit.getParams()
      trainingdp = getScaledTrainingData( examplefilenm, flatten=True,
                                          scaler=dgbkeys.globalstdtypestr,
                                          force=False,
                                          split=validation_split, nbfolds=None )
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
  except Exception as e:
    dgbmlio.announceTrainingFailure()
    raise e

def reformat( res, applyinfo ):
  """ For reformatting prediction result type(s)

  Parameters:
    * res (dict): predictions (labels, probabilities, confidence results)
    * applyinfo (dict): information from example file to apply model

  Returns:
    * dict: reformatted equivalence of results if key(s) match (labels, probabilities, confidence results)
  """

  if dgbkeys.preddictstr in res:
    res[dgbkeys.preddictstr] = res[dgbkeys.preddictstr].astype( applyinfo[dgbkeys.dtypepred] )
  if dgbkeys.probadictstr in res:
    res[dgbkeys.probadictstr] = res[dgbkeys.probadictstr].astype( applyinfo[dgbkeys.dtypeprob] )
  if dgbkeys.confdictstr in res:
    res[dgbkeys.confdictstr] = res[dgbkeys.confdictstr].astype( applyinfo[dgbkeys.dtypeconf] )
  return res

def doApplyFromFile( modelfnm, samples, outsubsel=None ):
  """
  """

  (model,info) = dgbmlio.getModel( modelfnm, fortrain=False )
  applyinfo = dgbmlio.getApplyInfo( info, outsubsel )
  return doApply( model, info, samples, applyinfo=applyinfo )

def doApply( model, info, samples, scaler=None, applyinfo=None, batchsize=None ):
  """ Applies a trained machine learning model on any platform for any workflow

  Parameters:
    * model (object): trained model in hdf5 format
    * info (dict): info from example file
    * samples (ndarray): input features to model
    * scaler (obj): scaler for scaling if any
    * applyinfo (dict): information from example file to apply model
    * batchsize (int): data batch size

  Returns:
    * dict: prediction results (reformatted, see dgbpy.mlapply.reformat)
  """

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
    inpshape = info[dgbkeys.inpshapedictstr]
    if isinstance(inpshape, int):
      inpshape = [inpshape]
    dictinpshape = tuple( inpshape )
    res = dgbkeras.apply( model, samples, isclassification, withpred, withprobs, withconfidence, doprobabilities, \
                          dictinpshape, scaler=None, batch_size=batchsize  )
  elif platform == dgbkeys.scikitplfnm:
    import dgbpy.dgbscikit as dgbscikit
    res = dgbscikit.apply( model, samples, scaler, isclassification, withpred, withprobs, withconfidence, doprobabilities )
  elif platform == dgbkeys.torchplfnm:
    import dgbpy.dgbtorch as dgbtorch
    res = dgbtorch.apply( model, info, samples, scaler, isclassification, withpred, withprobs, withconfidence, doprobabilities )
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
  """ Gets count of input images (train and validation)

  Parameters:
    * infos (dict): info from example file
    * raw (bool): set to True to return total input count,
                  False for otherwise (train and validation split counts)
    * dsets (dict): dataset

  Returns:
    * (dict, list): count of input images

  Notes:
    * a list of dictionary (train and validation input images counts) when
     raw=False. A dictionary of the total survey input images count
  """

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
  """ Gets count of input images

    * dsets (dict): dataset

  Returns:
    * (dict): count of total input images
  """
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
  np.random.shuffle( np.arange(np.int64(nrpts)) )
