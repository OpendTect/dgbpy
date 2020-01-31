#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : Arnaud
# DATE     : November 2018
#
# tools for reading hdf5 files for NN training
#

from os import path
import random
import json
import numpy as np
import h5py

import odpy.hdf5 as odhdf5
from odpy.common import std_msg

from dgbpy import dgbscikit
from dgbpy.keystr import *

hdf5ext = 'h5'

def getGroupNames( filenm ):
  h5file = h5py.File( filenm, 'r' )
  ret = list()
  for groupnm in h5file.keys():
    if isinstance( h5file[groupnm], h5py.Group ):
      ret.append( groupnm )
  h5file.close()
  return ret

def getInputNames( filenm ):
  h5file = h5py.File( filenm, 'r' )
  info = getInfo( filenm )
  ret = list(info[inputdictstr].keys())
  h5file.close()
  return ret

def getWellNms( examples, groupnm ):
  return examples[groupnm]['Wells'].keys()

def getNrGroups( filenm ):
  return len(getGroupNames(filenm))

def getNrInputs( filenm ):
  return len(getInputNames(filenm))

def getExampleInputs( dsets ):
  inps = list()
  for dsetnm in dsets:
    dset = dsets[dsetnm]
    for groupnm in dset:
      grp = dset[groupnm]
      for inpnm in grp:
        inps.append( (inpnm,groupnm) )
  ret = []
  for inp in inps:
    if inp not in ret:
      ret.append( inp )
  return ret

def getInputID( info, inpnm, groupnm ):
  if isLogOutput( info ):
    return info[exampledictstr][groupnm]['Wells'][inpnm]['id']
  return info[inputdictstr][inpnm]['id']

def getSubGroupKeys( info, inputs, groupnm ):
  if not isLogOutput( info ):
    return inputs
  return getWellNms( info[exampledictstr], groupnm )

def getCubeLetNames( info, groupnms, inputs ):
  ret = {}
  for groupnm in groupnms:
    inps = getSubGroupKeys( info, inputs, groupnm )
    ret.update({groupnm: getCubeLetNamesByGroup(info,inps,groupnm)} )
  return ret

def getCubeLetNamesByGroup( info, inputs, groupnm ):
  ret = {}
  for inp in inputs:
    ret.update({inp: getCubeLetNamesByGroupByInput(info,groupnm,inp)})
  return ret

def getCubeLetNamesByGroupByInput( info, groupnm, input ):
  h5file = h5py.File( info[filedictstr], 'r' )
  if not groupnm in h5file:
    return {}
  group = h5file[groupnm]
  dsetnms = list(group.keys())
  if xdatadictstr in dsetnms:
    ret = np.arange(len(group[xdatadictstr]))
  else:
    dsetwithinp = np.chararray.startswith( dsetnms, str(getInputID(info,input,groupnm))+':' )
    ret = np.extract( dsetwithinp, dsetnms )
  h5file.close()
  return np.ndarray.tolist(ret)

def getGroupSize( filenm, groupnm ):
  h5file = h5py.File( filenm, 'r' )
  group = h5file[groupnm]
  size = len(group)
  h5file.close()
  return size

def get_nr_attribs( info, subkey=None ):
  try:
    inputinfo = info[inputdictstr]
  except KeyError:
    raise
  ret = 0
  for groupnm in inputinfo:
    if subkey != None and groupnm != subkey:
      continue
    groupinfo = inputinfo[groupnm]
    try:
      nrattrib = len(groupinfo[attribdictstr])
    except KeyError:
      try:
        nrattrib = len(groupinfo[logdictstr])
      except KeyError:
        return 0
    if nrattrib == 0:
      continue
    if ret == 0:
      ret = nrattrib
    elif nrattrib != ret:
      raise ValueError
  return ret

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

def getNrOutputs( info ):
  return info[nroutdictstr]

def isSeisClass( info ):
  if isinstance(info,dict):
    return info[learntypedictstr] == seisclasstypestr
  return info == seisclasstypestr

def isLogOutput( info ):
  if isinstance(info,dict):
    return info[learntypedictstr] == loglogtypestr or info[learntypedictstr] == seisproptypestr
  return info == loglogtypestr or info == seisproptypestr

def isImg2Img( info ):
  if isinstance(info,dict):
    return info[learntypedictstr] == seisimgtoimgtypestr
  return info == seisimgtoimgtypestr

def getCubeLets( infos, datasets, groupnm ):
  survnm = groupnm.replace( ' ', '_' )
  fromwells = survnm in infos[inputdictstr]
  attribsel = None
  if fromwells:
    attribsel = survnm
  inpnrattribs = get_nr_attribs( infos, attribsel )
  inpshape = infos[inpshapedictstr]
  outshape = infos[outshapedictstr]
  nroutputs = getNrOutputs( infos )
  isclass = infos[classdictstr]
  img2img = isImg2Img( infos )
  if img2img:
    outnrattribs = 1
  outdtype = np.float32
  if isclass:
    outdtype = np.uint8
  h5file = h5py.File( infos[filedictstr], 'r' )
  group = h5file[groupnm]
  dsetnms = list(group.keys())
  hasdata = None
  if xdatadictstr in dsetnms and ydatadictstr in dsetnms:
    x_data = group[xdatadictstr]
    y_data = group[ydatadictstr]
    allcubelets = list()
    alloutputs = list()
    for inputnm in datasets:
      dsetnms = datasets[inputnm]
      nrpts = len(dsetnms)
      inpshape = get_np_shape(inpshape,nrpts,inpnrattribs)
      if img2img:
        outshape = get_np_shape(outshape,nrpts,outnrattribs)
      if len(x_data) == nrpts and len(y_data) == nrpts:
        cubelets = np.resize( x_data, inpshape ).astype( np.float32 )
        if img2img:
          output = np.resize( y_data, outshape ).astype( outdtype )
        else:
          output = np.resize( y_data, (nrpts,nroutputs) ).astype( outdtype )
      else:
        cubelets = np.empty( inpshape, np.float32 )
        if img2img:
          output = np.empty( outshape, outdtype )
        else:
          output = np.empty( (nrpts,nroutputs), outdtype )
        for idx,dsetnm in zip(range(len(dsetnms)),dsetnms):
          dset = x_data[dsetnm]
          odset = y_data[dsetnm]
          cubelets[idx] = np.resize( dset, cubelets[idx].shape )
          if img2img:
            output[idx] = np.resize( odset, output[idx].shape )
          else:
            output[idx] = np.asarray( odset )
      if nrpts > 0:
        allcubelets.append( cubelets )
        alloutputs.append( output )
    if len(allcubelets) > 0:
      cubelets = np.concatenate( allcubelets )
      hasdata = True
    if len(alloutputs) > 0:
      output = np.concatenate( alloutputs )
  else:
    allcubelets = list()
    alloutputs = list()
    for inputnm in datasets:
      dsetnms = datasets[inputnm]
      nrpts = len(dsetnms)
      inpshape = get_np_shape(inpshape,nrpts,inpnrattribs)
      cubelets = np.empty( inpshape, np.float32 )
      if img2img:
        outshape = get_np_shape(outshape,nrpts,outnrattribs)
        output = np.empty( outshape, outdtype )
      else:
        output = np.empty( (nrpts,nroutputs), outdtype )
      for idx,dsetnm in zip(range(len(dsetnms)),dsetnms):
        dset = group[dsetnm]
        if img2img:
          cubelets[idx] = np.resize(dset[1:],cubelets[idx].shape)
          output[idx] = np.resize(dset[0],output[idx].shape)
        else:
          cubelets[idx] = np.resize(dset,cubelets[idx].shape)
          if isclass :
            output[idx] = odhdf5.getIArray( dset, valuestr )
          else:
            output[idx] = odhdf5.getDArray( dset, valuestr )
      if nrpts > 0:
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

def getInfo( filenm ):
  h5file = h5py.File( filenm, 'r' )
  info = odhdf5.getInfoDataSet( h5file )
  if not validInfo( info ):
    h5file.close()
    return {}

  learntype = odhdf5.getText(info,typestr)
  isclassification = isSeisClass( learntype )
  if odhdf5.hasAttr(info,contentvalstr):
    isclassification = odhdf5.getText(info,contentvalstr) == classdatavalstr
  img2img = isImg2Img(learntype)
  logoutp = isLogOutput(learntype)

  extxt = 'Examples.'
  ex_sz = odhdf5.getIntValue(info, extxt+'Size') 
  idx = 0
  examples = {}
  while idx < ex_sz:
    exidxstr  = extxt+str(idx)+'.'
    example_sz = odhdf5.getIntValue( info, exidxstr+'Size' )
    example = list()
    idy = 0
    while idy < example_sz:
      exidystr = exidxstr+str(idy)+'.'
      exxyobj = {
        namedictstr: odhdf5.getText(info, exidystr+'Name' ),
        dbkeydictstr: odhdf5.getText(info, exidystr+'ID' ),
        iddictstr: idy,
      }
      exysurv = odhdf5.getText(info, exidystr+'Survey' )
      if len(exysurv) > 0:
        exxyobj.update({ locationdictstr: exysurv})
      example.append( exxyobj )
      idy += 1
    examples.update({ odhdf5.getText( info, exidxstr+'Name' ): {
      odhdf5.getText( info, exidxstr+'Type' ): example,
      targetdictstr: odhdf5.getText( info, exidxstr+'Target' ),
      iddictstr: idx
    }})
    idx += 1

  inptxt = 'Input.'
  inp_sz = odhdf5.getIntValue(info,inptxt+'Size')
  idx = 0
  inputs = {}
  while idx < inp_sz:
    inpidxstr = inptxt+str(idx)+'.'
    input_sz = odhdf5.getIntValue( info, inpidxstr+'Size' )
    input = list()
    idy = 0
    scales = list()
    means = list()
    while idy < input_sz:
      inpidystr = inpidxstr+str(idy)+'.'
      inpxyobj = {
        namedictstr: odhdf5.getText(info, inpidystr+'Name' ),
        dbkeydictstr: odhdf5.getText(info, inpidystr+'ID' ),
        iddictstr: idy
      }
      scalekey = inpidystr+'Stats'
      if odhdf5.hasAttr(info,scalekey):
        scaletxt = odhdf5.getAttr(info,scalekey)
        if scaletxt != '0`0':
          scale = odhdf5.getDInterval(info,scalekey)
          means.append( scale[0] )
          scales.append( scale[1] )
      input.append( inpxyobj )
      idy += 1
    inpobj = {
      attribdictstr: input,
      iddictstr: idx
    }
    inpsurv = odhdf5.getText(info, inpidxstr+'Survey' )
    if len(inpsurv) > 0:
      inpobj.update({ locationdictstr: inpsurv})
    if len(scales) > 0:
      inpobj.update({scaledictstr: dgbscikit.getNewScaler(means,scales) })
    inputs.update({ odhdf5.getText( info, inpidxstr+'Name' ): inpobj})
    idx += 1

  if not odhdf5.hasAttr(info,inpshapestr) or \
     not odhdf5.hasAttr(info,outshapestr):
     raise KeyError

  inpshape = odhdf5.getIArray( info, inpshapestr )
  outshape = odhdf5.getIArray( info, outshapestr )
  if logoutp:
    nroutputs = outshape
  else:
    nroutputs = 1

  retinfo = {
    learntypedictstr: learntype,
    inpshapedictstr: inpshape,
    outshapedictstr: outshape,
    nroutdictstr: nroutputs,
    classdictstr: isclassification,
    interpoldictstr: odhdf5.getBoolValue(info,"Edge extrapolation"),
    exampledictstr: examples,
    inputdictstr: inputs,
    filedictstr: filenm
  }

  retinfo.update({
    datasetdictstr: getCubeLetNames( retinfo, examples.keys(), inputs.keys() )
  })
  if odhdf5.hasAttr(info,'Model.Type' ):
    retinfo.update({plfdictstr: odhdf5.getText(info,'Model.Type')})
  if  odhdf5.hasAttr(info,versionstr):
    retinfo.update({versiondictstr: odhdf5.getText(info,versionstr)})
  h5file.close()

  if isLogOutput( learntype ):
    return getWellInfo( retinfo, filenm )
  elif isSeisClass(learntype) or isImg2Img(learntype):
    return getAttribInfo( retinfo, filenm )

  std_msg( "Unrecognized learn type: ", learntype )
  raise KeyError

def getAttribInfo( info, filenm ):
  if info[classdictstr]:
    if isSeisClass(info):
      info.update( {classesdictstr: getClassIndices(info)} )
    else:
      info.update( {classesdictstr: getClassIndicesFromData(info)} )

  info.update( {estimatedsizedictstr: getTotalSize(info)} )
  return info

def getNrClasses(info):
  return len(info[classesdictstr])

def arroneitemsize( dtype ):
  arr = np.empty(1,dtype)
  return arr.itemsize

def getTotalSize(info):
  inpnrattribs = get_nr_attribs( info )
  inpshape = info[inpshapedictstr]
  outshape = info[outshapedictstr]
  datasets = info[datasetdictstr]
  nrpts = 0
  for groupnm in datasets:
    alldata = datasets[groupnm]
    for inp in alldata:
      nrpts += len(alldata[inp])
  examplesshape = get_np_shape( inpshape, nrpts, inpnrattribs )
  x_size = np.prod( examplesshape ) * arroneitemsize( np.float32 )
  if info[classdictstr]:
    nroutvals = getNrClasses(info)
  else:
    nroutvals = getNrOutputs( info )
  outshape = get_np_shape( outshape, nrpts, nroutvals )
  y_size = np.prod( outshape ) * arroneitemsize( np.float32 )
  return x_size + y_size

def getWellInfo( info, filenm ):
  h5file = h5py.File( filenm, 'r' )
  infods = odhdf5.getInfoDataSet( h5file )
  info[classdictstr] = odhdf5.hasAttr(infods,'Target Value Type') and odhdf5.getText(infods,'Target Value Type') == "ID"
  if info[classdictstr]:
    info.update( {classesdictstr: getClassIndicesFromData(info)} )
  info.update( {estimatedsizedictstr: getTotalSize(info)} )
  zstep = odhdf5.getDValue(infods,"Z step") 
  marker = (odhdf5.getText(infods,"Top marker"),
            odhdf5.getText(infods,"Bottom marker"))
  h5file.close()
  info.update({
    zstepdictstr: zstep,
    rangedictstr: marker,
  })
  return info

modeloutstr = 'Model.Output.'
def modelIdxStr( idx ):
  return modeloutstr + str(idx) + '.Name'

def addInfo( inpfile, plfnm, filenm, infos=None ):
  if infos == None:
    infos = getInfo( inpfile )
  h5filein = h5py.File( inpfile, 'r' )
  h5fileout = h5py.File( filenm, 'r+' )
  dsinfoin = odhdf5.getInfoDataSet( h5filein )
  dsinfoout = odhdf5.ensureHasDataset( h5fileout )
  attribman = dsinfoin.attrs
  for attribkey in attribman:
    dsinfoout.attrs[attribkey] = attribman[attribkey]
  h5filein.close()
  odhdf5.setAttr( dsinfoout, versionstr, str(1) )
  odhdf5.setAttr( dsinfoout, 'Model.Type', plfnm )
  if plfnm == kerasplfnm:
    odhdf5.setArray( dsinfoout, inpshapestr, infos[inpshapedictstr] )
    if infos[learntypedictstr] == seisimgtoimgtypestr:
      odhdf5.setArray( dsinfoout, outshapestr, infos[outshapedictstr] )
    else:
      odhdf5.setAttr( dsinfoout, outshapestr, str(getNrOutputs(infos)))

  outps = getOutputs( inpfile )
  nrout = len(outps)
  odhdf5.setAttr( dsinfoout, modeloutstr+'Size', str(nrout) )
  for idx in range(nrout):
    odhdf5.setAttr( dsinfoout, modelIdxStr(idx), outps[idx] )

  inp = infos[inputdictstr]
  for inputnm in inp:
    input = inp[inputnm]
    if not scaledictstr in input:
      continue
    scale = input[scaledictstr]
    keyval = 'Input.' + str(input[iddictstr]) + '.Stats.'
    for i in range(len(scale.scale_)):
      odhdf5.setAttr( dsinfoout, keyval+str(i), str(scale.mean_[i])+'`'+str(scale.scale_[i]) )

  h5fileout.close()

def getClassIndices( info, filternms=None ):
  ret = []
  for groupnm in info[exampledictstr]:
    if filternms==None or groupnm in filternms:
      ret.append( info[exampledictstr][groupnm][iddictstr] )
  return np.sort( ret )

def getClassIndicesFromData( info ):
  if classesdictstr in info:
    return info[classesdictstr]
  filenm = info[filedictstr]
  h5file = h5py.File( filenm, 'r' )
  dsinfoin = odhdf5.ensureHasDataset( h5file )
  if odhdf5.hasAttr( dsinfoin, classesvalstr ):
    return odhdf5.getIArray( dsinfoin, classesvalstr )
  isimg2img =  isImg2Img( info )
  groups = getGroupNames( filenm )
  ret = list()
  for groupnm in groups:
    grp = h5file[groupnm]
    if isimg2img:
      for inpnm in grp:
        sublist = list(set(grp[inpnm][0].astype('uint8').ravel()))
        sublist.extend( ret )
        ret = list(set(sublist))
    else:
      nrvals = len(grp)
      vals = np.empty( nrvals, dtype='uint8' )
      for ival,dsetnm in zip(range(nrvals),grp):
        dset = grp[dsetnm]
        vals[ival] = odhdf5.getIntValue( dset, valuestr )
      sublist = list(set(vals.ravel()))
      sublist.extend( ret )
      ret = list(set(sublist))
  ret = np.sort( ret )
  h5file.close()
  h5fileout = h5py.File( filenm, 'r+' )
  dsinfoout = odhdf5.ensureHasDataset( h5fileout )
  odhdf5.setArray( dsinfoout, classesvalstr, ret )
  return ret
  
def getOutputs( inpfile ):
  info = getInfo( inpfile )
  ret = list()
  learntype = info[learntypedictstr]
  isclassification = info[classdictstr]
  if isclassification:
    ret.append( classificationvalstr  )
    if isSeisClass(learntype):
      ret.extend( getGroupNames(inpfile) )
      ret.append( confvalstr )
  elif isLogOutput(learntype) or isImg2Img(learntype):
    firsttarget = next(iter(info[exampledictstr]))
    targets = info[exampledictstr][firsttarget][targetdictstr]
    if isinstance(targets,list):
      ret.extend(targets)
    else:
      ret.append(targets)

  return ret

def getOutputNames( filenm, indices ):
  h5file = h5py.File( filenm, 'r' )
  info = odhdf5.getInfoDataSet( h5file )
  ret = list()
  for idx in indices:
    ret.append( odhdf5.getText(info,modelIdxStr(idx)) )
  h5file.close()
  return ret
