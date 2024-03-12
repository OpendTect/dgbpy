#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Nov 2018
#
# _________________________________________________________________________
# various tools machine learning data handling
#

import os
import numpy as np
from enum import Enum

import dgbpy.keystr as dgbkeys
import dgbpy.hdf5 as dgbhdf5
from odpy.common import restore_stdout, redirect_stdout

nladbdirid = '100060'
mlinpgrp = 'Deep Learning Example Data'
mltrlgrp = 'Deep Learning Model'
dgbtrl = 'dGB'

class StorageType(Enum):
  AWS = "AWS"
  LOCAL = "LOCAL"

def getInfo( filenm, quick=False ):
  """ Gets information from an example file

  Parameters:
    * filenm (str): ffile name/path in hdf5 format
    * quick (bool): when set to True, info is gottenn quickly leaving out some info(e.g. datasets), 
                    defaults to False and loads all informaton

  Returns:
    * dict: information from data file or model file                    
  """

  return dgbhdf5.getInfo( filenm, quick )

def datasetCount( dsets ):
  """Gets count of dataset

  Parameters:
    * dsets (dict): dataset

  Returns:
    * dict: counts for target attribute(s) or well(s) for  project
  """

  if dgbkeys.datasetdictstr in dsets:
    dsets = dsets[dgbkeys.datasetdictstr]
  ret = {}
  totsz = 0
  for groupnm in dsets:
    collection = dsets[groupnm]
    collcounts = {}
    groupsz = 0
    for collnm in collection:
      collitm = collection[collnm]
      collsz = len(collitm)
      groupsz += collsz
      collcounts.update({ collnm: collsz })
    collcounts.update({ 'size': groupsz })
    totsz += groupsz
    ret.update({groupnm: collcounts})
  ret.update({ 'size': totsz })
  return ret

def getDatasetNms( dsets, validation_split=None, valid_inputs=None ):
  """ Gets train and validation indices of dataset

  Parameters:
    * dsets (dict): dataset
    * validation_split (float): size of validation data (between 0-1)
    * valid_inputs (iter): 

  Returns:
    * dict: train and validation indices
  """

  train = {}
  valid = {}
  if validation_split == None:
    validation_split = 0
  dorandom = True
  if validation_split > 1:
    validation_split = 1
  elif validation_split < 0:
    validation_split = 0
  elif validation_split == 0:
    dorandom = False
  if valid_inputs == None:
    valid_inputs = {}
  for groupnm in dsets:
    group = dsets[groupnm]
    traingrp = {}
    validgrp = {}
    for inp in group:
      dsetnms = group[inp].copy()
      nrpts = len(dsetnms)
      if dorandom:
        nrpts = int(nrpts*validation_split)
        np.random.shuffle( dsetnms )
      if inp in valid_inputs:
        if dorandom:
          traingrp.update({inp: dsetnms[nrpts:]})
          validgrp.update({inp: dsetnms[:nrpts]})
        else:
          validgrp.update({inp: dsetnms})
      else:
        if dorandom and len(valid_inputs)<1:
          traingrp.update({inp: dsetnms[nrpts:]})
          validgrp.update({inp: dsetnms[:nrpts]})
        else:
          traingrp.update({inp: dsetnms})
    train.update({groupnm: traingrp})
    valid.update({groupnm: validgrp})
  return {
    dgbkeys.traindictstr: train,
    dgbkeys.validdictstr: valid
  }

def getCrossValidationIndices(dsets, seed=None, valid_inputs=1, nbfolds=5):
  """ Gets train and validation data for cross validation.

  Parameters:
      * dsets (dict): dictionary of survey names and datasets
      * n_wells (int): number of wells to use as the validat  ion set

  Returns:
      * list: list of dictionaries containing train and validation data for each fold
  """
  # Get all well names
  all_inp = [inp for groupnm in dsets.values() for inp in groupnm]
  if len(all_inp) == 1:
    return getDatasetNms(dsets, validation_split=valid_inputs)

  if valid_inputs == None:
    valid_inputs = 0
  if valid_inputs < 1 or valid_inputs > int(0.5*len(all_inp)):
    valid_inputs = int(0.20*len(all_inp)) #use 20% of number of wells as defaults

  # Shuffle well names
  np.random.seed(seed)
  np.random.shuffle(all_inp)

  # Initialize result list
  result = {}

  # Create folds
  for i in range(nbfolds):
    train, valid = {}, {}
    valid_wells = all_inp[i*valid_inputs : i*valid_inputs+valid_inputs]
    if len(valid_wells) < valid_inputs:
      break

    for groupnm in dsets:
      group = dsets[groupnm]
      traingrp = {}
      validgrp = {}
      for inp in group:
        dsetnms = group[inp].copy()
        np.random.shuffle(dsetnms)
        if inp in valid_wells:
          validgrp.update({inp: dsetnms})
        else:
          traingrp.update({inp: dsetnms})
      if traingrp:
        train.update({groupnm: traingrp})
      if validgrp:
        valid.update({groupnm: validgrp})
    result[dgbkeys.foldstr+f'{i+1}'] = {
      dgbkeys.traindictstr: train,
      dgbkeys.validdictstr: valid
      }
  return result

def getChunks(dsets,nbchunks):
  """ Splits dataset object into smaller chunks

  Parameters:
    * dsets (dict): dataset
    * nbchunks (int): number of data chunks to be created

  Returns:
    * dict: chunks from dataset stored as dictionaries
  """

  ret = []
  for ichunk in range(nbchunks):
    datagrp = {}
    for groupnm in dsets:
      alldata = dsets[groupnm]
      surveys = {}
      for inp in alldata:
        datalist = alldata[inp]
        nrpts = len(datalist)
        start = int(ichunk * nrpts / nbchunks)
        end = int((ichunk+1) * nrpts / nbchunks)
        datalist = datalist[start:end]
        surveys.update({inp: datalist})
      datagrp.update({groupnm: surveys})
    ret.append(datagrp)
  return ret
  
def hasScaler( infos, inputsel=None ):
  """ Checks if example file has scaleror not from info

  Parameters:
    * infos (dict): information about example file
    * inputsel (bool or NoneType):

  Returns:
    bool: True if dataset info has scaler key, False if other
  """

  inp = infos[dgbkeys.inputdictstr]
  for inputnm in inp:
    if inputsel != None and not inputnm in inputsel:
      continue
    if not dgbkeys.scaledictstr in inp[inputnm]:
      return False
  return True

def getDatasetsByGroup( dslist, groupnm ):
  ret = {}
  for keynm in dslist:
    dp = dslist[keynm]
    if groupnm in dp:
      ret.update({keynm:{groupnm: dp[groupnm]}})
  return ret

def getSomeDatasets( dslist, decim=None ):
  if decim == None or decim <= 0 or decim==False:
    return dslist
  if decim > 1:
    decim = 1
  ret = {}
  for dsetnm in dslist:
    sret = {}
    dset = dslist[dsetnm]
    for groupnm in dset:
      group = dset[groupnm]
      setgrp = {}
      if len(group) > 0 and isinstance(group[0],int):
        dsetnms = group.copy()
        nrpts = int(len(dsetnms)*decim)
        np.random.shuffle( dsetnms )
        del dsetnms[nrpts:]
        sret[groupnm] = dsetnms
      else:
        for inp in group:
          dsetnms = group[inp].copy()
          nrpts = int(len(dsetnms)*decim)
          np.random.shuffle( dsetnms )
          del dsetnms[nrpts:]
          setgrp[inp] = dsetnms
        sret[groupnm] = setgrp
    ret[dsetnm] = sret
  return ret

def getTrainingData( filenm, decim=False ):
  """ Gets training data from file name

  Parameters:
    * filenm (str): path to file in hdf5 format
    * decim (bool): 

  Returns:
    * dict: train, validation datasets as arrays, and info on example file
  """

  infos = getInfo( filenm )
  dsets = infos[dgbkeys.datasetdictstr]
  if decim:
      dsets = getSomeDatasets( dsets, decim )

  return getTrainingDataByInfo( infos, dsetsel=dsets )

def getTrainingDataByInfo( info, dsetsel=None ):
  """ Gets training data from file info

  Parameters:
    * info (dict): information about example file
    * dsetsel ():

  Returns:
    * dict: train, validation datasets as arrays, and info on example file
  """

  examples = dgbhdf5.getDatasets( info, dsetsel )
  ret = {}
  for ex in examples:
    ret.update({ex: examples[ex]})
  y_examples = list()
  if dgbkeys.ytraindictstr in examples:
    y_examples.append( examples[dgbkeys.ytraindictstr] )
  if dgbkeys.yvaliddictstr in examples:
    y_examples.append( examples[dgbkeys.yvaliddictstr] )
  ret.update({ dgbkeys.infodictstr: getClasses(info,y_examples) })
  if dgbkeys.classdictstr in info and info[dgbkeys.classdictstr]:
    if dgbkeys.ytraindictstr in examples:
      normalize_class_vector( examples[dgbkeys.ytraindictstr], \
                              info[dgbkeys.classesdictstr] )
    if dgbkeys.yvaliddictstr in examples:
      normalize_class_vector( examples[dgbkeys.yvaliddictstr], \
                              info[dgbkeys.classesdictstr] )
  return ret

def getClasses( info, y_vectors ):
  if not info[dgbkeys.classdictstr] or dgbkeys.classesdictstr in info:
    return info
  import numpy as np
  classes = []
  for y_vec in y_vectors:
    (minval,maxval) = ( np.min(y_vec), np.max(y_vec) )
    for idx in np.arange(minval,maxval+1,1,dtype=np.uint8):
      if np.any(y_vec == idx ):
        classes.append( idx )
  if len(classes) > 0:
    info.update( {dgbkeys.classesdictstr: np.array(classes,dtype=np.uint8)} )
  return info

def normalize_class_vector( arr, classes ):
  import numpy as np
  classes = np.sort( classes )
  for i in range( len(classes) ):
    arr[arr == classes[i]] = i

def unnormalize_class_vector( arr, classes ):
  import numpy as np
  classes = np.sort( classes )
  for i in reversed(range( len(classes) ) ):
    arr[arr == i] = classes[i]

def saveModel( model, inpfnm, platform, infos, outfnm, params ):
  """ Saves trained model for any platform workflow

  Parameters:
    * model (obj): trained model on any platform
    * inpfnm (str): example file name in hdf5 format
    * platform (str): machine learning platform (options; keras, Scikit-learn, torch)
    * infos (dict): example file info
    * outfnm (str): name of model to be saved
    * params (dict): parameters to be used when saving the model
  """

  from odpy.common import log_msg
  if not outfnm.endswith('.h5'):
    outfnm += '.h5'
  if os.path.exists(outfnm):
    try:
      os.remove( outfnm )
    except Exception as e:
      log_msg( '[Warning] Could not remove pre-existing model file:', e )
  log_msg( 'Saving model.' )
  if platform == dgbkeys.kerasplfnm:
    import dgbpy.dgbkeras as dgbkeras
    dgbkeras.save( model, outfnm )
  elif platform == dgbkeys.scikitplfnm:
    import dgbpy.dgbscikit as dgbscikit
    dgbscikit.save( model, outfnm )
  elif platform == dgbkeys.torchplfnm or platform == dgbkeys.onnxplfnm:
    import dgbpy.dgbtorch as dgbtorch
    dgbtorch.save( model, outfnm, infos, params )
  else:
    log_msg( 'Unsupported machine learning platform' )
    raise AttributeError
  dgbhdf5.addInfo( inpfnm, platform, outfnm, infos, model.__class__.__name__ )
  log_msg( 'Model saved.' )

def getModel( modelfnm, fortrain=False, pars=None ):
  """ Get model and model information

  Parameters:
    * modelfnm (str): model file path/name in hdf5 format
    * fortrain (bool): specifies if the model might be further trained
    * pars (dict): parameters to be used when restoring the model if needed

  Returs:
    * tuple: (trained model and model/project info)
  """

  infos = getInfo( modelfnm )
  platform = infos[dgbkeys.plfdictstr]
  if platform == dgbkeys.kerasplfnm:
    import dgbpy.dgbkeras as dgbkeras
    model = dgbkeras.load( modelfnm, fortrain, infos, pars )
  elif platform == dgbkeys.scikitplfnm:
    import dgbpy.dgbscikit as dgbscikit
    model = dgbscikit.load( modelfnm )
  elif platform == dgbkeys.torchplfnm:
    import dgbpy.dgbtorch as dgbtorch
    model = dgbtorch.load( modelfnm, infos )
  elif platform == dgbkeys.onnxplfnm:
    import dgbpy.dgbonnx as dgbonnx
    model = dgbonnx.load( modelfnm )
  else:
    from odpy.common import log_msg
    log_msg( 'Unsupported machine learning platform' )
    raise AttributeError
  return (model,infos)

def getApplyInfoFromFile( modelfnm, outsubsel=None ):
  """ Gets model apply info from file name

  Parameters:
    * modelfnm (str): model file path/name in hdf5 format
    * outsubsel ():

  Returns:
    * dict: apply information
  """

  return getApplyInfo( getInfo(modelfnm), outsubsel )

def getApplyInfo( infos, outsubsel=None ):
  """ Gets model apply info from example file info

  Parameters:
    * infos (dict): example file info
    * outsubsel ():

  Returns:
    * dict: apply information
  """

  isclassification = infos[dgbkeys.classdictstr]
  firstoutnm = dgbhdf5.getMainOutputs(infos)[0]
  if isclassification:
    names = firstoutnm
    preddtype = 'uint8'
  else:
    names = []
    preddtype = 'float32'

  probdtype = 'float32'
  confdtype = 'float32'
  if outsubsel != None:
    if 'targetnames' in outsubsel:
      names = outsubsel['targetnames']
    if dgbkeys.dtypepred in outsubsel:
      preddtype = outsubsel[dgbkeys.dtypepred]
    if dgbkeys.dtypeprob in outsubsel:
      probdtype = outsubsel[dgbkeys.dtypeprob]
    if dgbkeys.dtypeconf in outsubsel:
      confdtype = outsubsel[dgbkeys.dtypeconf]

  withpred = (isclassification and firstoutnm in names) or \
             not isclassification
  if isclassification:
    (withprobs,classnms) = dgbhdf5.getClassIndices( infos, names )
  else:
    withprobs = []
  ret = {
    dgbkeys.classdictstr: isclassification,
  }
  withconfidence = isclassification and dgbkeys.confvalstr in names

  if withpred:
    ret.update({dgbkeys.dtypepred: preddtype})
  if isclassification:
    if len(withprobs) > 0:
      ret.update({
        dgbkeys.probadictstr: withprobs,
        dgbkeys.dtypeprob: probdtype
      })
    if withconfidence:
      ret.update({dgbkeys.dtypeconf: confdtype})

  return ret

dblistall = None

def modelNameIsFree( modnm, type, args, reload=True ):
  (exists,sametrl,sameformat,sametyp) = \
             modelNameExists( modnm, type, args, reload )
  if exists == False:
    return True
  if not sametrl or not sameformat:
    return False
  if sametyp != None:
    return sametyp
  return False

def modelNameExists( modnm, type, args, reload=True ):
  (sametrl,sameformat,sametyp) = (None,None,None)
  modinfo = dbInfoForModel( modnm, args, reload )
  exists = modinfo != None
  if exists:
    sameformat = modinfo['Format'] == dgbtrl
    if 'Type' in modinfo:
      sametyp = type == modinfo['Type']
    if 'TranslatorGroup' in modinfo:
      sametrl = modinfo['TranslatorGroup'] == mltrlgrp
  return (exists,sametrl,sameformat,sametyp)

def dbInfoForModel( modnm, args, reload=True ):
  import odpy.dbman as oddbman
  global dblistall
  if dblistall == None or reload:
    dblistall = oddbman.getDBList(mltrlgrp, alltrlsgrps=True, args=args)
  return oddbman.getInfoFromDBListByNameOrKey( modnm, dblistall )

def getModelType( infos ):
  """ Gets model type

  Parameters:
    * infos (dict): example file info

  Returns:
    * str: Type ofmodel/workflow
  """

  return infos[dgbkeys.learntypedictstr]

def getSaveLoc( outnm, ftype, args ):
  try:
    import odpy.dbman as oddbman
    dblist = oddbman.getDBList(mltrlgrp,alltrlsgrps=False, args=args)
    try:
      dbkey = oddbman.getDBKeyForName( dblist, outnm )
    except ValueError:
      return oddbman.getNewEntryFileName(outnm,mltrlgrp,dgbtrl,\
                                        dgbhdf5.hdf5ext,ftype=ftype,args=args)
    return oddbman.getFileLocation( dbkey, args )
  except Exception:
    return os.path.join( os.getcwd(), outnm )
  
def announceShowTensorboard():
  restore_stdout()
  print('--ShowTensorboard--', flush=True)
  redirect_stdout()

def announceTrainingFailure():
  restore_stdout()
  print('--Training Failed--', flush=True)
  restore_stdout()  

def announceTrainingSuccess():
  restore_stdout()
  print('--Training Successful--', flush=True)
  restore_stdout()  
