#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : A. Huck
# DATE     : Nov 2018
#
# tools for reading hdf5 files for NN training
#

from os import path
import random
import json
import numpy as np

import odpy.hdf5 as odhdf5
from odpy.common import std_msg

from dgbpy import dgbscikit
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
  else:
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

def getNrOutputs( info ):
  return len( getMainOutputs(info) )

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

def isRegression( info ):
  return not isClassification( info )

def isClassification( info ):
  if instance(info, dict):
    return info[classdictstr]
  return info == classdatavalstr

def isSeisClass( info ):
  if isinstance(info,dict):
    return info[learntypedictstr] == seisclasstypestr
  return info == seisclasstypestr

def isLogInput( info ):
  if isinstance(info,dict):
    return info[learntypedictstr] == loglogtypestr
  return info == loglogtypestr

def isLogOutput( info ):
  if isinstance(info,dict):
    return info[learntypedictstr] == loglogtypestr or info[learntypedictstr] == seisproptypestr
  return info == loglogtypestr or info == seisproptypestr

def isImg2Img( info ):
  if isinstance(info,dict):
    return info[learntypedictstr] == seisimgtoimgtypestr
  return info == seisimgtoimgtypestr

def isModel( info ):
  return plfdictstr in info

def getCubeLets( infos, collection, groupnm ):
  if len(collection)< 1:
    return {}
  inpnrattribs = getNrAttribs( infos )
  inpshape = infos[inpshapedictstr]
  outshape = infos[outshapedictstr]
  nroutputs = getNrOutputs( infos )
  isclass = infos[classdictstr]
  img2img = isImg2Img( infos )
  examples = infos[exampledictstr]
  if img2img:
    outnrattribs = 1
  outdtype = np.float32
  if isclass:
    outdtype = np.uint8
  h5file = odhdf5.openFile( infos[filedictstr], 'r' )
  group = h5file[groupnm]
  dsetnms = list(group.keys())

  firstcollnm = next(iter(collection))
  hasdata = None
  if firstcollnm in group:
    allcubelets = list()
    alloutputs = list()
    for collnm in collection:
      x_data = group[collnm][xdatadictstr]
      y_data = group[collnm][ydatadictstr]
      dsetnms = collection[collnm]
      nrpts = len(dsetnms)
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
    for collnm in collection:
      dsetnms = collection[collnm]
      nrpts = len(dsetnms)
      inparrshape = get_np_shape(inpshape,nrpts,inpnrattribs)
      cubelets = np.empty( inparrshape, np.float32 )
      if img2img:
        outarrshape = get_np_shape(outshape,nrpts,outnrattribs)
        output = np.empty( outarrshape, outdtype )
      else:
        output = np.empty( (nrpts,nroutputs), outdtype )
      for idx,dsetnm in zip(range(len(dsetnms)),dsetnms):
        dset = group[dsetnm]
        if img2img:
          try:
            cubelets[idx] = np.resize(dset[:-1],cubelets[idx].shape)
            output[idx] = np.resize(dset[-1],output[idx].shape)
          except Exception as e:
            cubelets[idx] = np.zeros( cubelets[idx].shape, cubelets.dtype )
            output[idx] = np.zeros( output[idx].shape, output[idx].dtype )
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

def getInfo( filenm, quick ):
  h5file = odhdf5.openFile( filenm, 'r' )
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
    exxobj = {
      targetdictstr: odhdf5.getText( info, exidxstr+'Target' ),
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
      collnm = odhdf5.getText(info, inpidystr+'Name' )
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
      inpobj.update({scaledictstr: dgbscikit.getNewScaler(means,scales) })
    inputs.update({ odhdf5.getText( info, inpidxstr+'Name' ): inpobj})
    idx += 1

  if not odhdf5.hasAttr(info,inpshapestr) or \
     not odhdf5.hasAttr(info,outshapestr):
     raise KeyError

  inpshape = odhdf5.getIArray( info, inpshapestr )
  outshape = odhdf5.getIArray( info, outshapestr )

  retinfo = {
    learntypedictstr: learntype,
    inpshapedictstr: inpshape,
    outshapedictstr: outshape,
    classdictstr: isclassification,
    interpoldictstr: odhdf5.getBoolValue(info,"Edge extrapolation"),
    exampledictstr: examples,
    inputdictstr: inputs,
    filedictstr: filenm
  }

  if not quick:
    retinfo.update({
      datasetdictstr: getCubeLetNames( retinfo )
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
  return len(info[classesdictstr])

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
  isimg2img = isImg2Img( info )
  groups = info[exampledictstr].keys()
  ret = list()
  for groupnm in groups:
    grp = h5file[groupnm]
    if isimg2img:
      for inpnm in grp:
        sublist = list(set(grp[inpnm][-1].astype('uint8').ravel()))
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
