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
import re
import sys

import dgbpy.keystr as dgbkeys
import dgbpy.hdf5 as dgbhdf5
from odpy.common import restore_stdout, redirect_stdout, log_msg

nladbdirid = '100060'
mlinpgrp = 'Deep Learning Example Data'
mltrlgrp = 'Deep Learning Model'
dgbtrl = 'dGB'

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

def getDatasetNms( dsets, validation_split=None, valid_inputs=None, seed=None ):
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
        np.random.seed(seed)
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

def getCrossValidationIndices(dsets, valid_inputs=1, nbfolds=5, seed=None):
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
  checksets = [ dgbkeys.xtraindictstr, dgbkeys.xvaliddictstr ]
  # Clustering data has no target to validate
  if not dgbhdf5.isSegmentation( info ):
    checksets += [ dgbkeys.ytraindictstr, dgbkeys.yvaliddictstr ]
  for setnm in checksets:
    if setnm in examples:
      checkFiniteData( examples[setnm], setnm )
  ret.update({ dgbkeys.infodictstr: getClasses(info,y_examples) })
  if dgbhdf5.isClassification( info ):
    if dgbkeys.ytraindictstr in examples:
      normalize_class_vector( examples[dgbkeys.ytraindictstr], \
                              info[dgbkeys.classesdictstr] )
      checkClassVector( examples[dgbkeys.ytraindictstr],
                        info[dgbkeys.classesdictstr], dgbkeys.ytraindictstr )
    if dgbkeys.yvaliddictstr in examples:
      normalize_class_vector( examples[dgbkeys.yvaliddictstr], \
                              info[dgbkeys.classesdictstr] )
      checkClassVector( examples[dgbkeys.yvaliddictstr],
                        info[dgbkeys.classesdictstr], dgbkeys.yvaliddictstr )
  return ret

def getClasses( info, y_vectors ):
  if not dgbhdf5.isClassification( info ):
    return info
  import numpy as np
  # The class list covers every group of the example file; callers pass one
  # group at a time, so re-deriving it here narrowed the list and left targets
  # out of range for CrossEntropyLoss. Derive only when none is given.
  if dgbkeys.classesdictstr in info and len(info[dgbkeys.classesdictstr]) > 0:
    return info
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

# OpendTect's undefined float. Undefined or non-finite training data gives a
# NaN loss and a model of NaN weights, so reject it up front with a clear message.
odundefval = 1e30

def checkFiniteData( arr, setnm ):
  """Rejects NaN, inf and OpendTect undefined values before they reach a loss."""
  import numpy as np
  if arr is None or not isinstance( arr, np.ndarray ) or arr.size < 1:
    return
  if not np.issubdtype( arr.dtype, np.floating ):
    return
  nonfinite = ~np.isfinite( arr )
  nrnonfinite = int( np.count_nonzero( nonfinite ) )
  nrundef = int( np.count_nonzero( np.abs(arr[~nonfinite]) >= odundefval/10 ) )
  if nrnonfinite < 1 and nrundef < 1:
    return
  what = []
  if nrundef > 0:
    what.append( '%d OpendTect undefined (1e30)' % nrundef )
  if nrnonfinite > 0:
    what.append( '%d NaN or infinite' % nrnonfinite )
  raise ValueError(
    '%s holds %s of %d values (%.2f%%). Training on these produces a NaN loss '
    'and a model of NaN weights. Remove the undefined samples from the '
    'training set, or restrict the training area to where every input and '
    'target is defined.'
    % (setnm, ' and '.join(what), arr.size,
       100.0*(nrnonfinite+nrundef)/arr.size) )

def checkClassVector( arr, classes, setnm ):
  """Rejects class values outside range(len(classes)) before CrossEntropyLoss
  trips an opaque device-side assert on them."""
  import numpy as np
  nrclasses = len( classes )
  if nrclasses < 1 or arr.size < 1:
    return
  bad = np.unique( arr[(arr < 0) | (arr >= nrclasses)] )
  if len(bad) < 1:
    return
  raise ValueError(
    'Class values %s in %s fall outside the %d classes of the example file '
    '(%s): the training data holds values the example file does not declare '
    'as classes.' % (list(bad), setnm, nrclasses, list(classes)) )

def unnormalize_class_vector( arr, classes ):
  import numpy as np
  classes = np.sort( classes )
  for i in reversed(range( len(classes) ) ):
    arr[arr == i] = classes[i]

def saveModel( model, inpfnm, platform, infos, outfnm, params, **kwargs ):
  """ Saves trained model for any platform workflow

  Parameters:
    * model (obj): trained model on any platform
    * inpfnm (str): example file name in hdf5 format
    * platform (str): machine learning platform (options; keras, Scikit-learn, torch)
    * infos (dict): example file info
    * outfnm (str): name of model to be saved or S3 folder URI
    * params (dict): parameters to be used when saving the model
  """
  isbokeh = kwargs.get('isbokeh', False)
  if dgbhdf5.shouldUseS3(outfnm, params, kwargs=kwargs):
    import dgbpy.dgb_boto as dgb_boto
    save_function = lambda modelfnm: saveModel(model, inpfnm, platform, infos, modelfnm, params, isHandled=True)
    return dgb_boto.handleS3FileSaving(save_function, outfnm, params, isbokeh=isbokeh)

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
    if dgbscikit.isClustering(model):
      savetype = params['savetype']
    else:
      savetype = dgbscikit.defsavetype
    dgbscikit.save( model, outfnm, savetype )
  elif platform == dgbkeys.torchplfnm or platform == dgbkeys.onnxplfnm:
    import dgbpy.dgbtorch as dgbtorch
    dgbtorch.save( model, outfnm, infos, params )
  else:
    log_msg( 'Unsupported machine learning platform' )
    raise AttributeError
  dgbhdf5.addInfo( inpfnm, platform, outfnm, infos, model.__class__.__name__ )
  log_msg( 'Model saved.' )

def getModel( modelfnm, fortrain=False, pars=None, **kwargs ):
  """ Get model and model information

  Parameters:
    * modelfnm (str): model file path/name in hdf5 format
    * fortrain (bool): specifies if the model might be further trained
    * pars (dict): parameters to be used when restoring the model if needed

  Returs:
    * tuple: (trained model and model/project info)
  """
  if dgbhdf5.shouldUseS3(modelfnm, params=None, kwargs=kwargs):
    import dgbpy.dgb_boto as dgb_boto
    load_function = lambda s3uri: getModel(s3uri, fortrain, pars, isHandled=True)
    return dgb_boto.handleS3FileLoading(load_function, modelfnm)

  infos = getInfo( modelfnm )
  platform = infos[dgbkeys.plfdictstr]
  if dgbhdf5.isZipModel(infos):
    import dgbpy.zipmodelbase as dgbzipmodel
    model = dgbzipmodel.load(modelfnm)
  elif platform == dgbkeys.kerasplfnm:
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
  """ Gets model apply info from example/model file name

  Parameters:
    * modelfnm (str): model file path/name in hdf5 format
    * outsubsel ():

  Returns:
    * dict: apply information
  """

  return getApplyInfo( getInfo(modelfnm), outsubsel )

def getApplyInfo( infos, outsubsel=None ):
  """ Gets model apply info from example/model file info

  Parameters:
    * infos (dict): example file info
    * outsubsel ():

  Returns:
    * dict: apply information
  """

  isclassification = dgbhdf5.isClassification( infos )
  firstoutnm = dgbhdf5.getMainOutputs(infos)[0]
  if isclassification:
    names = firstoutnm
    if dgbhdf5.isImg2Img(infos):
      preddtype = 'float32'
    else:
      classinfo = np.array( infos[dgbkeys.classesdictstr] )
      hasunlabels = dgbhdf5.hasUnlabeled( infos )
      preddtype = dgbhdf5.getOutdType( classinfo, hasunlabels ).__name__
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

  ret = {}
  if isclassification:
    ret[dgbkeys.classdictstr] = isclassification

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

# ---------------------------------------------------------------------------
# OpendTect database access, out of process
#
# Importing odpy.dbman loads odbind and with it ODHDF5, which binds to the
# hdf5.dll h5py already loaded. When the two HDF5 versions differ this corrupts
# the shared HDF5 state and the model save fails afterwards, typically with
#   KeyError("Property object doesn't exist")  or  ValueError("Not a datatype")
# The database calls only need the .omf metadata, so run them in a child
# process. Set DGBPY_DB_INPROCESS=1 for the old in-process behaviour.
# Remove once ODHDF5 is linked statically against HDF5.
# ---------------------------------------------------------------------------

_dbjsonmark = '__ODDBJSON__'

# The request goes over stdin: sys.path can exceed the Windows command line limit.
_dbbootstrap = (
  'import sys, json;'
  'req = json.loads(sys.stdin.read());'
  'sys.path[:0] = req["path"];'
  'import dgbpy.mlio as _m;'
  'sys.stdout.write("' + _dbjsonmark + '" + json.dumps('
  '_m.dbCallDirect(req["op"], req["payload"])))'
)


def dbSafeArgs( args ):
  """ Keeps only the JSON-serialisable survey-location entries of args """

  import json
  ret = {}
  if not args:
    return ret
  for ky in ('dtectdata','survey','dtectexec'):
    if ky not in args:
      continue
    try:
      json.dumps( args[ky] )
      ret[ky] = args[ky]
    except Exception:
      pass
  return ret


def dbCallDirect( op, payload ):
  """ Performs an OpendTect database call in this process """

  import odpy.dbman as oddbman
  args = payload.get( 'args', {} ) or {}
  if op == 'getSaveLoc':
    dblist = oddbman.getDBList( mltrlgrp, alltrlsgrps=False, args=args )
    dbkey = None
    try:
      dbkey = oddbman.getDBKeyForName( dblist, payload['outnm'] )
    except ValueError:
      dbkey = None
    if not dbkey:
      # getDBKeyForName returns None for an unknown name
      return oddbman.getNewEntryFileName( payload['outnm'], mltrlgrp, dgbtrl,
                                          args=args )
    return oddbman.getFileLocation( dbkey, args )
  if op == 'dbInfoForModel':
    dblistall_ = oddbman.getDBList( mltrlgrp, alltrlsgrps=True, args=args )
    return oddbman.getInfoFromDBListByNameOrKey( payload['modnm'], dblistall_ )
  raise ValueError( 'Unknown database operation: ' + str(op) )


def dbCall( op, payload ):
  """ Runs an OpendTect database call, in a child process where possible """

  if os.environ.get( 'DGBPY_DB_INPROCESS' ) == '1':
    return dbCallDirect( op, payload )
  import json, subprocess
  try:
    # args can hold open file objects (argparse.FileType): send only the
    # survey-location keys
    payload = dict( payload )
    payload['args'] = dbSafeArgs( payload.get('args') )
    req = json.dumps( {'path': list(sys.path), 'op': op, 'payload': payload} )
    proc = subprocess.run(
             [sys.executable, '-c', _dbbootstrap],
             input=req.encode('utf-8'),
             stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300 )
    out = proc.stdout.decode( 'utf-8', 'replace' )
    idx = out.rfind( _dbjsonmark )
    if idx >= 0:
      return json.loads( out[idx+len(_dbjsonmark):] )
    log_msg( '[Warning] Out-of-process database call for', op,
             'returned no result (exit code', str(proc.returncode) + ').',
             'Falling back in-process, which loads ODHDF5 and will very likely',
             'break the save. Child stderr follows:' )
    log_msg( proc.stderr.decode('utf-8','replace')[-1500:] )
  except Exception as e:
    log_msg( '[Warning] Out-of-process database call failed:', repr(e),
             '- falling back in-process.' )
  return dbCallDirect( op, payload )


def dbInfoForModel( modnm, args, reload=True ):
  global dblistall
  return dbCall( 'dbInfoForModel', {'modnm': modnm, 'args': args} )

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
    return dbCall( 'getSaveLoc',
                   {'outnm': outnm, 'ftype': ftype, 'args': args} )
  except Exception:
    return os.path.join( os.getcwd(), outnm )
  
def getStoredParams(uiparams, params):
  tabs = {
          "training": [dgbkeys.stoptrainkeystr],
          "parameters": [dgbkeys.typekeystr, dgbkeys.splitkeystr, dgbkeys.batchkeystr,
                        dgbkeys.epochskeystr, dgbkeys.patiencekeystr, dgbkeys.learnratekeystr,
                        dgbkeys.epochdropkeystr, dgbkeys.decimkeystr, dgbkeys.prefercpustr,
                        dgbkeys.userandomseeddictstr],
          "advanced": [dgbkeys.scaledictstr, dgbkeys.transformkeystr, dgbkeys.tofp16keystr,
                      dgbkeys.withtensorboardkeystr, dgbkeys.savetypekeystr]
          }

  for tab, keys in tabs.items():
    tab_params = params.get(tab, {})
    for key in keys:
      value = tab_params.get(key)
      if isinstance(value, str) and value.lower() == 'null':
        value = None
      if isinstance(value, str) and value.lower() in ['true', 'false']:
        value = value.lower() == 'true'
      if value is not None:
        uiparams[key] = value
  
  return uiparams

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

def load_libstdcpp( odpid ):
  maps_path = '/proc/'+str(odpid)+'/maps'
  if not os.path.exists(maps_path):
    return

  odlibstdcpppath = None
  with open(maps_path) as f:
    for line in f:
      if "libstdc++" in line:
        m = re.search(r"/.*libstdc\+\+.*", line)
        if m:
          odlibstdcpppath = m.group(0)
          break

  if odlibstdcpppath is None:
    return

  odlibstdcppver = int(odlibstdcpppath.rsplit('.', 1)[-1])

  python_path = sys.executable
  pythonbin_dir = os.path.dirname(python_path)
  pythonroot_dir = os.path.dirname(pythonbin_dir)
  pythonlib_dir = os.path.join(pythonroot_dir, "lib")
  if not os.path.exists(pythonlib_dir):
    return

  candidate = os.path.join(pythonlib_dir, 'libstdc++.so')
  if not os.path.exists(candidate):
    return

  pythonlibstdcpppath = os.path.realpath(candidate)
  pythonlibstdcppver = int(pythonlibstdcpppath.rsplit('.', 1)[-1])
  if odlibstdcppver > pythonlibstdcppver:
    import odbind # force early load of dependent libraries
