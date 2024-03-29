#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : A. Huck
# DATE     : Nov 2018
#
# tools for reading hdf5 files for NN training
#
import os
import random
import json
import ast
from datetime import datetime
from enum import Enum

import numpy as np
from pathlib import PurePosixPath, PureWindowsPath, Path
from odpy.common import log_msg,  redirect_stdout, restore_stdout, isWin

import odpy.hdf5 as odhdf5
from odpy.common import std_msg

from dgbpy.keystr import *

hdf5ext = 'h5'

def dictAddIfNew( newset, toadd ):
  ret = toadd
  for itmnm in newset:
    toadd.update({itmnm: newset[itmnm]})
  return ret

def getCubeLetNames( info ):
  examples = info[exampledictstr]
  ret = {}
  for groupnm in examples:
    example = examples[groupnm]
    ret.update({groupnm: getCubeLetNamesByGroup(info,groupnm,example)} )
  return ret

def getCubeLetNamesByGroup( info, groupnm, example ):
  collection = example[collectdictstr]
  ret = {}
  for collnm in collection:
    itmidx = collection[collnm][iddictstr]
    ret.update({collnm: getCubeLetNamesByGroupByItem(info,groupnm,collnm,itmidx)})
  return ret

def getCubeLetNamesByGroupByItem( info, groupnm, collnm, idx ):
  h5file = odhdf5.openFile( info[filedictstr], 'r' )
  if not groupnm in h5file:
    return {}
  group = h5file[groupnm]
  dsetnms = list(group.keys())
  if collnm in dsetnms:
    ret = np.arange(len(group[collnm][xdatadictstr]))
  elif len(dsetnms)==1 and collnm in dsetnms[0].split('`'):  # img2img multi-target
    ret = np.arange(len(group[dsetnms[0]][xdatadictstr]))
  else:
    dsetnms = list(map(str, dsetnms))
    dsetwithinp = np.chararray.startswith( dsetnms, str(idx)+':' )
    ret = np.extract( dsetwithinp, dsetnms )
  h5file.close()
  return np.ndarray.tolist(ret)

def getGroupSize( filenm, groupnm ):
  h5file = odhdf5.openFile( filenm, 'r' )
  group = h5file[groupnm]
  size = len(group)
  h5file.close()
  return size

def getNrAttribs( info ):
  input = info[inputdictstr]
  collection = input[next(iter(input))][collectdictstr]
  return len(collection)

def getNrGroupInputs( info ):
  inps = []
  examples = info[exampledictstr]
  for groupnm in examples:
    if collectdictstr in examples[groupnm]:
      collection = examples[groupnm][collectdictstr]
      for inp in collection:
        inps.append(inp)
  return len(inps)

def getNrOutputs( info ):
  return len( getMainOutputs(info) )

def getSeed( info ):
  if isinstance(info, dict) and seeddictstr in info:
    return info[seeddictstr]

def get_np_shape( shape, nrpts=None, nrattribs=None ):
  ret = ()
  if nrpts != None:
    ret += (nrpts,)
  if nrattribs != None:
    ret += (nrattribs,)
  if isinstance( shape, int ):
    ret += ( 1,1,shape, )
    return ret
  for i in shape:
    ret += (i,)
  return ret

def getTrainingConfig( h5file ):
  if trainconfigdictstr in h5file.attrs:
    config = h5file.attrs[trainconfigdictstr]
    return json.loads(config)
  return {}

def isRegression( info ):
  return not isClassification( info )

def isClassification( info ):
  if isinstance(info, dict):
    return info[classdictstr]
  return info == classdatavalstr

def isSegmentation( info ):
  if isinstance(info,dict):
    return info[segmentdictstr]
  return info == segmenttypestr

def isSeisClass( info ):
  if isinstance(info,dict):
    return info[learntypedictstr] == seisclasstypestr
  return info == seisclasstypestr

def hasUnlabeled( info ):
  if isinstance(info, dict):
    return info[withunlabeleddictstr]
  return info == withunlabeleddictstr

def isLogInput( info ):
  if isinstance(info,dict):
    return info[learntypedictstr] == loglogtypestr
  return info == loglogtypestr

def isLogOutput( info ):
  if isinstance(info,dict):
    return info[learntypedictstr] == loglogtypestr or \
           info[learntypedictstr] == seisproptypestr or \
           info[learntypedictstr] == logclustertypestr
  return info == loglogtypestr or info == seisproptypestr or info == logclustertypestr

def isImg2Img( info ):
  if isinstance(info,dict):
    return info[learntypedictstr] == seisimgtoimgtypestr
  return info == seisimgtoimgtypestr

def isCrossValidation( info ):
  if isLogInput(info):
    if trainseldicstr in info:
      found_fold = any(foldstr in chunk for chunk in info[trainseldicstr][0])
      return found_fold
    if exampledictstr in info:
      return getNrGroupInputs(info) > 1
  return False

def unscaleOutput( info ):
  if isinstance(info,dict) and outputunscaledictstr in info:
    return info[outputunscaledictstr]
  return False

def applyGlobalStd( info ):
  if isinstance(info,dict) and inpscalingdictstr in info:
    return info[inpscalingdictstr] == globalstdtypestr
  return info == globalstdtypestr

def applyLocalStd( info ):
  if isinstance(info,dict) and inpscalingdictstr in info:
    return info[inpscalingdictstr] == localstdtypestr
  return info == localstdtypestr

def applyNormalization( info ):
  if isinstance(info,dict) and inpscalingdictstr in info:
    return info[inpscalingdictstr] == normalizetypestr
  return info == normalizetypestr

def applyMinMaxScaling( info ):
  if isinstance(info,dict) and inpscalingdictstr in info:
    return info[inpscalingdictstr] == minmaxtypestr
  return info == minmaxtypestr

def applyArrTranspose( info ):
  if isinstance(info,dict) and arrayorderdictstr in info:
    return info[arrayorderdictstr] == reversestr
  return info == reversestr

class StorageType(Enum):
  AWS = "AWS"
  LOCAL = "LOCAL"

class Scaler(Enum):
  GlobalScaler = globalstdtypestr
  StandardScaler = localstdtypestr
  Normalization = normalizetypestr
  MinMaxScaler = minmaxtypestr

def isDefaultScaler(scaler, info, uselearntype=True):
  _isLogOutput = isLogOutput(info) and uselearntype
  if _isLogOutput or scaler == globalstdtypestr:
    return scaler, True
  return scaler, False

def updateScaleInfo( scaler, info ):
  if not scaler:
    return info
  info[inpscalingdictstr] = scaler
  info[outputunscaledictstr] = doOutputScaling(info)
  return info

def getScalerStr( info ):
  if isinstance(info,dict):
    if inpscalingdictstr in info:
      return info[inpscalingdictstr]
    return globalstdtypestr
  return info == inpscalingdictstr 

def doOutputScaling( info ):
  if isImg2Img(info) and isRegression(info):
    return True
  return False

def isModel( info ):
  return plfdictstr in info

def isMultiLabelRegression( info ):
  if not isRegression( info ) or isImg2Img( info ):
    return False
  return getNrOutputs( info ) > 1

def hasboto3(auth=False):
    try:
        import boto3
        if auth:
            s3 = boto3.client('s3')
            s3.list_buckets()
        return True
    except Exception as e:
        return False

def isS3Uri(uri):
    return uri.startswith('s3://')

def shouldUseS3(modelfnm, params=None, relaxed=True, kwargs=None):   
  if not hasboto3(not relaxed):
    return
  if not isS3Uri(modelfnm) and params and params.get('storagetype') != StorageType.AWS.value:
    return
  if kwargs and kwargs.get('isHandled'):
    return
  return True
  
def rm_tree(pth):
    pth = Path(pth)
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            try:
              rm_tree(child)
            except OSError:
              pass
    try:
      pth.rmdir()
    except OSError:
      pass

def getLogDir( withtensorboard, examplenm, platform, basedir, clearlogs, args ):
  if not withtensorboard or basedir == None or not Path(basedir).exists():
    return None
  examplenm = platform+'_'+Path(examplenm).stem
  logdir = Path(basedir, examplenm)
  if logdir.exists():
      if clearlogs:
         for child in logdir.glob('*'):
            rm_tree(child)
  else:
      try:
         logdir.mkdir()
      except:
         return None

  if surveydictstr in args:
    jobnm = args[surveydictstr][0] + '_run'
  else:
    jobnm = 'run'

  nrsavedruns = 0
  with os.scandir(logdir) as it:
    for entry in logdir.iterdir():
      if entry.name.startswith(jobnm) and entry.is_dir():
        nrsavedruns += 1
  logdir = logdir / Path(jobnm+str(nrsavedruns+1)+'_'+'m'.join( datetime.now().isoformat().split(':')[:-1] ))
  return logdir

def getOutdType( classinfo, hasunlabels=False ):
  max = classinfo.max()
  min = classinfo.min()
  min_abs = np.abs(min)
  max_abs = np.abs(max)
  if hasunlabels:
    min = np.min([min, -1])
  if max_abs>min_abs:
    pass
  else:
    max_abs = min_abs
  if min < 0 and max > 0:
    if max_abs < 129:
      return np.int8
    elif max_abs > 128 and max_abs < 32769:
      return np.int16
    elif max_abs > 32768 and max_abs < int(2.2e10):
      return np.int32
    else:
      return np.int64
  else:
    if max_abs < 256:
      return np.uint8
    elif max_abs > 255 and max_abs < int(6.5e4):
      return np.uint16
    elif max_abs > int(6.5e4) and max_abs < int(4.4e10):
      return np.uint32
    else:
      return np.uint64

def getCubeLets_img2img_multitarget( infos, collection, groupnm ):
  inpnrattribs = getNrAttribs( infos )
  outnrattribs = getNrOutputs( infos )
  inpshape = infos[inpshapedictstr]
  outshape = infos[outshapedictstr]
  examples = infos[exampledictstr]
  h5file = odhdf5.openFile( infos[filedictstr], 'r' )
  outdtype = np.float32
  group = h5file[groupnm]
  targetnm = '`'.join([colnm for colnm in collection])
  if (not targetnm in group) or (not xdatadictstr in group[targetnm]) or (not ydatadictstr in group[targetnm]):
    return {}

  x_data = group[targetnm][xdatadictstr]
  y_data = group[targetnm][ydatadictstr]
  dsetnms = next(iter(collection.values()))
  nrpts = len(dsetnms)
  if nrpts < 1:
    return {}
    
  inparrshape = get_np_shape(inpshape,nrpts,inpnrattribs)
  outarrshape = get_np_shape(outshape,nrpts,outnrattribs)
  inputs = None
  outputs = None
  if len(x_data) == nrpts and len(y_data) == nrpts:
    inputs = np.resize( x_data, inparrshape ).astype( np.float32 )
    outputs = np.resize( y_data, outarrshape ).astype( outdtype )
  else:
    inputs = np.empty( inparrshape, np.float32 )
    outputs = np.empty( outarrshape, outdtype )
    for idx,dsetnm in zip(range(len(dsetnms)),dsetnms):
      dset = x_data[dsetnm]
      odset = y_data[dsetnm]
      inputs[idx] = np.resize( dset, inputs[idx].shape )
      outputs[idx] = np.resize( odset, outputs[idx].shape )

  hasdata = len(inputs)>0 and len(outputs)>0
  h5file.close()
  return  {
            xtraindictstr: inputs,
            ytraindictstr: outputs
          } if hasdata else {}

def getCubeLets( infos, collection, groupnm ):
  if len(collection)< 1:
    return {}
  img2img = isImg2Img( infos )
  nroutputs = getNrOutputs( infos )
  if img2img and nroutputs>1:
    return getCubeLets_img2img_multitarget( infos, collection, groupnm )

  inpnrattribs = getNrAttribs( infos )
  inpshape = infos[inpshapedictstr]
  outshape = infos[outshapedictstr]
  isclass = infos[classdictstr]
  iscluster = isSegmentation( infos )
  examples = infos[exampledictstr]
  h5file = odhdf5.openFile( infos[filedictstr], 'r' )
  if img2img:
    outnrattribs = 1
  outdtype = np.float32
  if isclass:
    outdtype = getOutdType(np.array(infos[classesdictstr]), hasUnlabeled( infos ))
  group = h5file[groupnm]

  firstcollnm = next(iter(collection))
  hasdata = None
  allcubelets = list()
  alloutputs = list()
  for collnm in collection:
    if not collnm in group:
      continue

    grp = group[collnm]
    if not xdatadictstr in grp:
      continue

    if not iscluster and not ydatadictstr in grp:
      continue

    x_data = grp[xdatadictstr]
    if not iscluster:
      y_data = grp[ydatadictstr]
    else:
      y_data = []

    dsetnms = collection[collnm]
    nrpts = len(dsetnms)
    if nrpts < 1:
      continue

    hasydata = len(y_data) > 0
    inparrshape = get_np_shape(inpshape,nrpts,inpnrattribs)
    if img2img:
      outarrshape = get_np_shape(outshape,nrpts,outnrattribs)
    if len(x_data) == nrpts and len(y_data) == nrpts:
      cubelets = np.resize( x_data, inparrshape ).astype( np.float32 )
      if img2img:
        output = np.resize( y_data, outarrshape ).astype( outdtype )
      else:
        output = np.resize( y_data, (nrpts,nroutputs) ).astype( outdtype )
    else:
      cubelets = np.empty( inparrshape, np.float32 )
      if img2img:
        output = np.empty( outarrshape, outdtype )
      else:
        output = np.empty( (nrpts,nroutputs), outdtype )
      for idx,dsetnm in zip(range(len(dsetnms)),dsetnms):
        dset = x_data[dsetnm]
        if hasydata:
          odset = y_data[dsetnm]

        cubelets[idx] = np.resize( dset, cubelets[idx].shape )
        if hasydata:
          if img2img:
            output[idx] = np.resize( odset, output[idx].shape )
          else:
            output[idx] = np.asarray( odset )

    allcubelets.append( cubelets )
    alloutputs.append( output )

  if len(allcubelets) > 0:
    cubelets = np.concatenate( allcubelets )
    hasdata = True
  if len(alloutputs) > 0:
    output = np.concatenate( alloutputs )
  h5file.close()
  if not hasdata:
    return {}
  return {
    xtraindictstr: cubelets,
    ytraindictstr: output
  }

def getDatasets_( infos, datasets, fortrain ):
  dictkeys = list()
  if fortrain:
    dictkeys.append( xtraindictstr )
    dictkeys.append( ytraindictstr )
  else:
    dictkeys.append( xvaliddictstr )
    dictkeys.append( yvaliddictstr )
  ret = {}
  cubelets = list()
  for groupnm in datasets:
    cubes = getCubeLets(infos,datasets[groupnm],groupnm)
    if len(cubes) > 0:
      cubelets.append( cubes )
  allx = list()
  ally = list()
  for cubelet in cubelets:
    allx.append( cubelet[xtraindictstr] )
    ally.append( cubelet[ytraindictstr] )
  if len(allx) > 0:
    ret.update({dictkeys[0]: np.concatenate( allx )})
    ret.update({dictkeys[1]: np.concatenate( ally )})
  return ret

def getDatasets( infos, dsetsel=None, train=True, validation=True ):
  ret = {}
  if dsetsel == None:
    datasets = infos[datasetdictstr]
  else:
    datasets = dsetsel
  if train:
    if traindictstr in datasets:
      traindsets = datasets[traindictstr]
    else:
      traindsets = datasets
    trainret = getDatasets_( infos, traindsets, True )
    if len(trainret) > 0:
      for ex in trainret:
        ret.update({ex: trainret[ex]})
  if validation and validdictstr in datasets:
    validret = getDatasets_( infos, datasets[validdictstr], False )
    if len(validret) > 0:
      for ex in validret:
        ret.update({ex: validret[ex]})
  return ret

def validInfo( info ):
  try:
    learntype = odhdf5.getText(info,typestr)
  except KeyError:
    std_msg("No type found. Probably wrong type of hdf5 file")
    return False
  return True

def getInfo( filenm, quick ):
  h5file = odhdf5.openFile( filenm, 'r' )
  info = odhdf5.getInfoDataSet( h5file )
  if not validInfo( info ):
    h5file.close()
    return {}

  modname = None
  if odhdf5.hasAttr(info, modelnmstr):
    modname = odhdf5.getText(info, modelnmstr)

  learntype = odhdf5.getText(info,typestr)
  isclassification = isSeisClass( learntype )
  issegmentation = False
  if odhdf5.hasAttr(info,contentvalstr):
    isclassification = odhdf5.getText(info,contentvalstr) == classdatavalstr
    issegmentation = odhdf5.getText(info,contentvalstr) == segmenttypestr
  img2img = isImg2Img(learntype)
  logoutp = isLogOutput(learntype)

  hasunlabels = False
  if odhdf5.hasAttr(info, withunlabeleddictstr):
    hasunlabels = odhdf5.getBoolValue( info, withunlabeleddictstr )

  arrayorder = carrorderstr
  arrorderstr = 'Examples.ArrayOrder'
  if odhdf5.hasAttr(info,arrorderstr):
    arrayorder = odhdf5.getText(info,arrorderstr)

  scalingtype = globalstdtypestr
  scalingtypestr='Input.Scaling.Type'
  if odhdf5.hasAttr(info,scalingtypestr):
    scalingtype = odhdf5.getText(info,scalingtypestr)

  scalingvalrg = [0,255]
  scalingvalstr = 'Input.Scaling.Value Range'
  if odhdf5.hasAttr(info,scalingvalstr):
    scalingvalrg = odhdf5.getDInterval(info,scalingvalstr)

  unscaleoutput = False
  outunscalestr = 'Output.Unscale'
  if odhdf5.hasAttr(info,outunscalestr):
    unscaleoutput = odhdf5.getBoolValue(info,outunscalestr)

  extxt = 'Examples.'
  ex_sz = odhdf5.getIntValue(info, extxt+'Size')
  idx = 0
  examples = {}
  while idx < ex_sz:
    exidxstr  = extxt+str(idx)+'.'
    targetnmstr = exidxstr+'Target'
    example_sz = odhdf5.getIntValue( info, exidxstr+'Size' )
    targetnms = odhdf5.getText( info, targetnmstr)
    exxobj = {
      targetdictstr: targetnms,
      iddictstr: idx
    }
    exsurv = odhdf5.getText(info, exidxstr+'Survey' )
    if len(exsurv) > 0:
      exxobj.update({ locationdictstr: exsurv})
    excompnrstr = exidxstr+'Component'
    if odhdf5.hasAttr(info,excompnrstr):
      exxobj.update({componentdictstr: odhdf5.getIntValue(info,excompnrstr)})
    collection = {}
    idy = 0
    while idy < example_sz:
      exidystr = exidxstr+str(idy)+'.'
      collnm = odhdf5.getText(info,exidystr+'Name')
      collnm = collnm.replace( '/', '_' )
      exxyobj = {
        dbkeydictstr: odhdf5.getText(info, exidystr+'ID' ),
        iddictstr: idy,
      }
      classnmstr = exidystr+'Class Name'
      if odhdf5.hasAttr(info,classnmstr):
        exxyobj.update({classnmdictstr: odhdf5.getText(info,classnmstr)})
      gidstr = exidystr+'GeomID'
      if odhdf5.hasAttr(info,gidstr):
        exxyobj.update({geomiddictstr: odhdf5.getIntValue(info,gidstr)})
      linenmstr = exidystr+'Line name'
      if odhdf5.hasAttr(info,linenmstr):
        collnm = odhdf5.getText(info,linenmstr)
        if len(collnm) < 1:
          idy = idy+1
          continue
      collection.update({ collnm: exxyobj })
      idy += 1
    if len(collection) > 0:
      exxobj.update({collectdictstr: collection})
    examples.update({ odhdf5.getText(info,exidxstr+'Name'): exxobj })
    idx += 1

  inptxt = 'Input.'
  inp_sz = odhdf5.getIntValue(info,inptxt+'Size')
  idx = 0
  inputs = {}
  while idx < inp_sz:
    inpidxstr = inptxt+str(idx)+'.'
    input_sz = odhdf5.getIntValue( info, inpidxstr+'Size' )
    collection = {}
    idy = 0
    scales = list()
    means = list()
    while idy < input_sz:
      inpidystr = inpidxstr+str(idy)+'.'
      collnm = odhdf5.getAttr(info, inpidystr+'Name' )
      inpxyobj = {
        iddictstr: idy
      }
      compkey = inpidystr+'Component'
      if odhdf5.hasAttr(info,compkey):
        collnm += '`' + odhdf5.getText(info,compkey)
      idkey = inpidystr+'ID'
      if odhdf5.hasAttr(info,idkey):
        inpxyobj.update({dbkeydictstr: odhdf5.getText(info,idkey)})
      scalekey = inpidystr+'Stats'
      if odhdf5.hasAttr(info,scalekey):
        scaletxt = odhdf5.getAttr(info,scalekey)
        if scaletxt != '0`0':
          scale = odhdf5.getDInterval(info,scalekey)
          means.append( scale[0] )
          scales.append( scale[1] )
      collection.update({ collnm: inpxyobj })
      idy += 1
    inpobj = {
      collectdictstr: collection,
      iddictstr: idx
    }
    inpsurv = odhdf5.getText(info, inpidxstr+'Survey' )
    if len(inpsurv) > 0:
      inpobj.update({ locationdictstr: inpsurv})
    if len(scales) > 0:
      from dgbpy import dgbscikit
      inpobj.update({scaledictstr: dgbscikit.getNewScaler(means,scales) })
    inputs.update({ odhdf5.getText( info, inpidxstr+'Name' ): inpobj})
    idx += 1

  if not odhdf5.hasAttr(info,inpshapestr) or \
     not odhdf5.hasAttr(info,outshapestr):
     raise KeyError

  inpshape = odhdf5.getIArray( info, inpshapestr )
  outshape = odhdf5.getIArray( info, outshapestr )

  retinfo = {
    namedictstr: modname,
    learntypedictstr: learntype,
    segmentdictstr: issegmentation,
    inpshapedictstr: inpshape,
    outshapedictstr: outshape,
    classdictstr: isclassification,
    interpoldictstr: odhdf5.getBoolValue(info,"Edge extrapolation"),
    arrayorderdictstr: arrayorder,
    inpscalingdictstr: scalingtype,
    inpscalingvalsdictstr: scalingvalrg,
    outputunscaledictstr: unscaleoutput,
    exampledictstr: examples,
    inputdictstr: inputs,
    filedictstr: filenm,
    withunlabeleddictstr: hasunlabels
  }

  if not quick:
    retinfo.update({
      datasetdictstr: getCubeLetNames( retinfo )
      })

  if odhdf5.hasAttr(info,'Model.Type' ):
    retinfo.update({plfdictstr: odhdf5.getText(info,'Model.Type')})
  if  odhdf5.hasAttr(info,versionstr):
    retinfo.update({versiondictstr: odhdf5.getText(info,versionstr)})
  trainingconfig = getTrainingConfig( h5file )
  retinfo.update({trainconfigdictstr: trainingconfig})
  h5file.close()

  if isLogOutput(learntype):
    return getWellInfo( retinfo, filenm )

  if isSeisClass(learntype) or isImg2Img(learntype):
    return getAttribInfo( retinfo, filenm )

  std_msg( "Unrecognized learn type: ", learntype )
  raise KeyError

def getAttribInfo( info, filenm ):
  if info[classdictstr]:
    if isSeisClass(info):
      (classidxs,classnms) = getClassIndices(info)
      classidxs = np.array(classidxs)+1
      info.update( {
        classesdictstr: classidxs.tolist(),
        classnmdictstr: classnms
      })
    else:
      info.update( {classesdictstr: getClassIndicesFromData(info)} )

  if not isModel(info):
    info.update( {estimatedsizedictstr: getTotalSize(info)} )
  return info

def getWellInfo( info, filenm ):
  if info[classdictstr]:
    info.update( {
      classesdictstr: getClassIndicesFromData(info),
      classnmdictstr: getMainOutputs(info)[0]
    })
  else:
    info[outshapedictstr] = getNrOutputs(info)

  if not isModel(info):
    info.update( {estimatedsizedictstr: getTotalSize(info)} )
  h5file = odhdf5.openFile( filenm, 'r' )
  infods = odhdf5.getInfoDataSet( h5file )
  zstep = odhdf5.getDValue(infods,"Z step")
  marker = (odhdf5.getText(infods,"Top marker"),
            odhdf5.getText(infods,"Bottom marker"))
  h5file.close()
  info.update({
    zstepdictstr: zstep,
    rangedictstr: marker,
  })
  return info

def getNrClasses(info):
  if classesdictstr in info:
    return len(info[classesdictstr])
  return 1

def arroneitemsize( dtype ):
  arr = np.empty(1,dtype)
  return arr.itemsize

def getTotalSize( info ):
  inpnrattribs = getNrAttribs( info )
  inpshape = info[inpshapedictstr]
  outshape = info[outshapedictstr]
  examples = info[exampledictstr]
  h5file = odhdf5.openFile( info[filedictstr], 'r' )
  nrpts = 0
  for groupnm in examples:
    grp = h5file[groupnm]
    collection = examples[groupnm][collectdictstr]
    firstcollnm = next(iter(collection))
    if not firstcollnm in grp:
      nrpts += len(grp)
      continue
    for collnm in collection:
      nrpts += len(grp[collnm][xdatadictstr])
  h5file.close()
  examplesshape = get_np_shape( inpshape, nrpts, inpnrattribs )
  x_size = np.prod( examplesshape, dtype=np.int64 ) * arroneitemsize( np.float32 )
  if info[classdictstr]:
    nroutvals = getNrClasses( info )
  else:
    nroutvals = getNrOutputs( info )
  outshape = get_np_shape( outshape, nrpts, nroutvals )
  y_size = np.prod( outshape, dtype=np.int64 ) * arroneitemsize( np.float32 )
  return x_size + y_size

modeloutstr = 'Model.Output.'
def modelIdxStr( idx ):
  return modeloutstr + str(idx) + '.Name'

def odsetBoolValue(value):
  if value:
    return 'Yes'
  return 'No'

def addInfo( inpfile, plfnm, filenm, infos, clssnm ):
  h5filein = odhdf5.openFile( inpfile, 'r' )
  h5fileout = odhdf5.openFile( filenm, 'r+' )
  dsinfoin = odhdf5.getInfoDataSet( h5filein )
  dsinfoout = odhdf5.ensureHasDataset( h5fileout )
  attribman = dsinfoin.attrs
  for attribkey in attribman:
    dsinfoout.attrs[attribkey] = attribman[attribkey]
  h5filein.close()
  odhdf5.setAttr( dsinfoout, versionstr, str(1) )
  odhdf5.setAttr( dsinfoout, 'Model.Type', plfnm )
  odhdf5.setAttr( dsinfoout, 'Model.Class', clssnm )
  if plfnm == kerasplfnm:
    #The model size may be smaller than from the example data
    odhdf5.setArray( dsinfoout, inpshapestr, infos[inpshapedictstr] )
    if isImg2Img( infos ):
      odhdf5.setArray( dsinfoout, outshapestr, infos[outshapedictstr] )
    else:
      odhdf5.setArray( dsinfoout, outshapestr, getNrOutputs(infos) )

  outps = getOutputs( infos )
  odhdf5.setAttr( dsinfoout, modeloutstr+'Size', str(len(outps)) )
  for idx,outp in zip(range(len(outps)),outps):
    odhdf5.setAttr( dsinfoout, modelIdxStr(idx), outp )

  inp = infos[inputdictstr]
  for inputnm in inp:
    input = inp[inputnm]
    if not scaledictstr in input:
      continue
    scale = input[scaledictstr]
    keyval = 'Input.' + str(input[iddictstr]) + '.'
    for i in range(len(scale.scale_)):
      keyvali = keyval+str(i)+'.Stats'
      odhdf5.setArray( dsinfoout, keyvali, [scale.mean_[i], scale.scale_[i]] )
  odhdf5.setAttr(dsinfoout, 'Input.Scaling.Type', infos[inpscalingdictstr])
  odhdf5.setAttr(dsinfoout, 'Output.Unscale', odsetBoolValue(infos[outputunscaledictstr]))
  h5fileout.close()

def getClassIndices( info, filternms=None ):
  if isLogOutput( info ):
    return ([],[]) #No support (yet?)
  allclasses = {}
  examples = info[exampledictstr]
  for groupnm in examples:
    collection = examples[groupnm][collectdictstr]
    collclassnms = {}
    for collnm in collection:
      collitm = collection[collnm]
      if not classnmdictstr in collitm:
        continue
      classnm = collitm[classnmdictstr]
      if filternms==None or classnm in filternms:
        collclassnms.update({classnm: collitm[iddictstr]})
    dictAddIfNew( collclassnms, allclasses )
  idxs = list()
  classnms = list()
  for classnm in allclasses:
    classitm = allclasses[classnm]
    idxs.append( classitm )
    classnms.append( classnm )
  return (idxs,classnms)

def getClassIndicesFromData( info ):
  if classesdictstr in info:
    return info[classesdictstr]
  filenm = info[filedictstr]
  h5file = odhdf5.openFile( filenm, 'r' )
  dsinfoin = odhdf5.ensureHasDataset( h5file )
  if odhdf5.hasAttr( dsinfoin, classesvalstr ):
    return odhdf5.getIArray( dsinfoin, classesvalstr )
  groups = info[exampledictstr].keys()
  ret = list()
  for groupnm in groups:
    grp = h5file[groupnm]
    for inpnm in grp:
      outdtype = getOutdType(np.array(grp[inpnm][ydatadictstr]), hasUnlabeled( info ))
      sublist = list(set(np.array(grp[inpnm][ydatadictstr]).astype(outdtype).ravel()))
      sublist.extend( ret )
      ret = list(set(sublist))
  ret = np.sort( ret )
  h5file.close()
  h5fileout = odhdf5.openFile( filenm, 'r+' )
  dsinfoout = odhdf5.ensureHasDataset( h5fileout )
  odhdf5.setArray( dsinfoout, classesvalstr, ret )
  h5fileout.close()
  return ret

def getMainOutputs( info ):
  examples = info[exampledictstr]
  firstexamplenm = next(iter(examples))
  targets = examples[firstexamplenm][targetdictstr]
  ret = list()
  if isinstance(targets,list):
    ret.extend(targets)
  else:
    ret.append(targets)
  return ret

def getOutputs( info ):
  ret = getMainOutputs( info )
  if isSeisClass(info):
    ret.extend( info[classnmdictstr] )
    ret.append( confvalstr )
  return ret

def getOutputNames( filenm, indices ):
  h5file = odhdf5.openFile( filenm, 'r' )
  info = odhdf5.getInfoDataSet( h5file )
  ret = list()
  for idx in indices:
    ret.append( odhdf5.getText(info,modelIdxStr(idx)) )
  h5file.close()
  return ret

def translateFnm( modfnm, modelfnm ):
  posidxh5fp = PurePosixPath( modelfnm )
  winh5fp = PureWindowsPath( modelfnm )
  posixmodfp = PurePosixPath( modfnm )
  winmodfp = PureWindowsPath( modfnm )
  import os
  if isWin():
    moddir = winh5fp.parent
    modbasefnm = winmodfp.name
    modlocfnm = PureWindowsPath( moddir ).joinpath( PureWindowsPath(modbasefnm))
    relmodlocfnm = modlocfnm.with_name( winh5fp.name )
    relmodlocfnm = relmodlocfnm.with_suffix( winmodfp.suffix )
    if os.path.exists(relmodlocfnm):
      modfnm = relmodlocfnm
    elif os.path.exists(modlocfnm):
      modfnm = modlocfnm
    else:
      moddir = posidxh5fp.parent
      modbasefnm = posixmodfp.name
      modlocfnm = PurePosixPath( moddir ).joinpath( PurePosixPath(modbasefnm) )
      relmodlocfnm = modlocfnm.with_name( posidxh5fp.name )
      relmodlocfnm = relmodlocfnm.with_suffix( posixmodfp.suffix )
      if os.path.exists(relmodlocfnm):
        modfnm = relmodlocfnm        
      elif os.path.exists(modlocfnm):
        modfnm = modlocfnm
  else:
    moddir = posidxh5fp.parent
    modbasefnm = posixmodfp.name
    modlocfnm = PurePosixPath( moddir ).joinpath( PurePosixPath(modbasefnm) )
    relmodlocfnm = modlocfnm.with_name( posidxh5fp.name )
    relmodlocfnm = relmodlocfnm.with_suffix( posixmodfp.suffix )
    if os.path.exists(relmodlocfnm):
      modfnm = relmodlocfnm
    elif os.path.exists(modlocfnm):
      modfnm = modlocfnm
    else:
      moddir = winh5fp.parent
      modbasefnm = winmodfp.name
      modlocfnm = PureWindowsPath( moddir ).joinpath(PureWindowsPath(modbasefnm))
      relmodlocfnm = modlocfnm.with_name( winh5fp.name )
      relmodlocfnm = relmodlocfnm.with_suffix( winmodfp.suffix )
      modlocfnm = modlocfnm.as_posix()
      relmodlocfnm = relmodlocfnm.as_posix()
      if os.path.exists(relmodlocfnm):
        modfnm = relmodlocfnm
      elif os.path.exists(modlocfnm):
        modfnm = modlocfnm
  return modfnm
