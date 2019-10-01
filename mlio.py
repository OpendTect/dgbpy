#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Nov 2018
#
# _________________________________________________________________________
# various tools machine learning data handling
#

import numpy as np

import odpy.dbman as oddbman
import dgbpy.keystr as dgbkeys
import dgbpy.hdf5 as dgbhdf5

nladbdirid = '100060'
mltrlgrp = 'Deep Learning Model'
dgbtrl = 'dGB'

def getInfo( filenm ):
  return dgbhdf5.getInfo( filenm )

def getDatasetNms( dsets, validation_split=None, valid_inputs=None ):
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

def hasScaler( infos, inputsel=None ):
  inp = infos[dgbkeys.inputdictstr]
  for inputnm in inp:
    if inputsel != None and not inputnm in inputsel:
      continue
    if not dgbkeys.scaledictstr in inp[inputnm]:
      return False
  return True

def getDatasetsByInput( dslist, inp ):
  ret = {}
  for dslistnm in dslist:
    retdset = {}
    dsets = dslist[dslistnm]
    for groupnm in dsets:
      group = dsets[groupnm]
      retgrp = {}
      if inp in group:
        retgrp.update({inp: group[inp]})
      retdset.update({groupnm: retgrp})
    ret.update({dslistnm: retdset})
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
  infos = getInfo( filenm )
  return getTrainingDataByInfo( infos )

def getTrainingDataByInfo( info, dsetsel=None ):
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

def saveModel( model, inpfnm, platform, infos, outfnm ):
  from odpy.common import log_msg
  log_msg( 'Saving model.' )
  if platform == dgbkeys.kerasplfnm:
    import dgbpy.dgbkeras as dgbkeras
    dgbkeras.save( model, outfnm )
  elif platform == dgbkeys.scikitplfnm:
    import dgbpy.dgbscikit as dgbscikit
    log_msg( 'Unsupported machine learning platform' )
    raise AttributeError
  else:
    log_msg( 'Unsupported machine learning platform' )
    raise AttributeError
  dgbhdf5.addInfo( inpfnm, platform, outfnm, infos )
  log_msg( 'Model saved.' )

def getModel( modelfnm ):
  infos = dgbhdf5.getInfo( modelfnm )
  platform = infos[dgbkeys.plfdictstr]
  if platform == dgbkeys.kerasplfnm:
    import dgbpy.dgbkeras as dgbkeras
    model = dgbkeras.load( modelfnm )
  elif platform == dgbkeys.scikitplfnm:
    import dgbpy.dgbscikit as dgbscikit
    model = dgbscikit.load( modelfnm )
  else:
    from odpy.common import log_msg
    log_msg( 'Unsupported machine learning platform' )
    raise AttributeError
  return (model,infos)

def getApplyInfoFromFile( modelfnm, outsubsel=None ):
  return getApplyInfo( dgbhdf5.getInfo(modelfnm), outsubsel )

def getApplyInfo( infos, outsubsel=None ):
  isclassification = infos[dgbkeys.classdictstr]
  if isclassification:
    names = [dgbkeys.classvalstr]
    preddtype = 'uint8'
  else:
    names = []
    preddtype = 'float32'

  probdtype = 'float32'
  confdtype = 'float32'
  if outsubsel != None:
    if 'names' in outsubsel:
      names = outsubsel['names']
    if dgbkeys.dtypepred in outsubsel:
      preddtype = outsubsel[dgbkeys.dtypepred]
    if dgbkeys.dtypeprob in outsubsel:
      probdtype = outsubsel[dgbkeys.dtypeprob]
    if dgbkeys.dtypeconf in outsubsel:
      confdtype = outsubsel[dgbkeys.dtypeconf]

  withpred = (isclassification and dgbkeys.classvalstr in names) or \
             not isclassification
  if isclassification:
    withprobs = dgbhdf5.getClassIndices( infos, names )
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

def getSaveLoc( outnm, args ):
  dblist = oddbman.getDBList(mltrlgrp,args)
  try:
    dbkey = oddbman.getDBKeyForName( dblist, outnm )
  except ValueError:
    return oddbman.getNewEntryFileName(outnm,nladbdirid,mltrlgrp,dgbtrl,\
                                       dgbhdf5.hdf5ext,args)
  return oddbman.getFileLocation( dbkey, args )
