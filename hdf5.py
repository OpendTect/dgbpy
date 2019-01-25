#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : Arnaud
# DATE     : November 2018
#
# tools for reading hdf5 files for NN training
#

from os import path
import json
import numpy as np
import h5py
import odpy.hdf5 as odhdf5
from odpy.common import std_msg
from dgbpy.keystr import *

hdf5ext = 'h5'

def getGroupNames( filenm ):
  h5file = h5py.File( filenm, "r" )
  ret = list()
  for groupnm in h5file.keys():
    if isinstance( h5file[groupnm], h5py.Group ):
      ret.append( groupnm )
  h5file.close()
  return ret

def getNrGroups( filenm ):
  return len(getGroupNames(filenm))

def getGroupSize( filenm, groupnm ):
  h5file = h5py.File( filenm, "r" )
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

def get_np_shape( step, nrpts=None, nrattribs=None ):
  ret = ()
  if nrpts != None:
    ret += (nrpts,)
  if nrattribs != None:
    ret += (nrattribs,)
  if isinstance( step, int ):
    ret += ( step*2+1, )
    return ret
  for i in step:
    ret += (i*2+1,)
  return ret

def getCubeLets( filenm, infos, groupnm, decim ):
  fromwells = groupnm in infos[inputdictstr]
  attribsel = None
  if fromwells:
    attribsel = groupnm
  nrattribs = get_nr_attribs( infos, attribsel )
  stepout = infos[stepoutdictstr]
  isclass = infos[classdictstr]
  if decim:
    if decim < 0 or decim > 100:
      std_msg( "Decimation percentage not within [0,100]" )
      raise ValueError
  h5file = h5py.File( filenm, "r" )
  group = h5file[groupnm]
  dsetnms = list(group.keys())
  nrpts = len(dsetnms)
  if decim:
    np.random.shuffle( dsetnms )
    nrpts = int(nrpts*(decim/100))
    if nrpts < 1:
      return {}
    del dsetnms[nrpts:]
  shape = None
  if fromwells :
    if stepout > 0:
      shape = ( nrpts, nrattribs, stepout*2+1 )
    else:
      shape = ( nrpts, nrattribs )
  else:
    shape = get_np_shape(stepout,nrpts,nrattribs)

  cubelets = np.empty( shape, np.float32 )
  outdtype = np.float32
  if isclass:
    outdtype = np.uint8
  output = np.empty( nrpts, outdtype )
  idx = 0
  for dsetnm in dsetnms:
    dset = group[dsetnm]
    cubelets[idx] = np.array( dset )
    if isclass :
      output[idx] = odhdf5.getIntValue( dset, valuestr )
    else:
      output[idx] = odhdf5.getDValue( dset, valuestr )
    idx += 1

  h5file.close()
  ret = {
    xtraindictstr: cubelets,
    ytraindictstr: output
  }
  return ret

def getAllCubeLets( filenm, decim=False ):
  infos = getInfo( filenm )
  groupnms = getGroupNames( filenm )
  cubelets = list()
  for groupnm in groupnms:
    cubelets.append( getCubeLets(filenm,infos,groupnm,decim) )
  totsz = 0
  allx = list()
  ally = list()
  for cubelet in cubelets:
    totsz += len(cubelet[xtraindictstr])
    allx.append( cubelet[xtraindictstr] )
    ally.append( cubelet[ytraindictstr] )
  return {
    xtraindictstr: np.concatenate( allx ),
    ytraindictstr: np.concatenate( ally )
  }

def validInfo( info ):
  try:
    type = odhdf5.getText(info,typestr)
  except KeyError:
    std_msg("No type found. Probably wrong type of hdf5 file")
    return False
  return True

def getInfo( filenm ):
  h5file = h5py.File( filenm, "r" )
  info = odhdf5.getInfoDataSet( h5file )
  if not validInfo( info ):
    h5file.close()
    return {}

  type = odhdf5.getText(info,typestr)
  if odhdf5.hasAttr(info,"Trace.Stepout"):
    stepout = odhdf5.getIStepInterval(info,"Trace.Stepout") 
  elif odhdf5.hasAttr(info,"Depth.Stepout"):
    stepout = odhdf5.getIStepInterval(info,"Depth.Stepout")
  classification = True
  ex_sz = odhdf5.getIntValue(info,"Examples.Size") 
  idx = 0
  examples = {}
  while idx < ex_sz:
    namestr = "Examples."+str(idx)+".Name"
    logstr = "Examples."+str(idx)+".Log"
    if odhdf5.hasAttr( info, namestr ):
      exname = namestr
      extype = "Point-Sets"
    elif odhdf5.hasAttr( info, logstr ):
      exname = logstr
      extype = "Logs"
    else:
      raise KeyError
    grouplbl = odhdf5.getText( info, exname )
    example = {}
    example_sz = odhdf5.getIntValue(info,"Examples."+str(idx)+".Size")
    idy = 0
    while idy < example_sz:
      exyname = odhdf5.getText(info,"Examples."+str(idx)+".Name."+str(idy))
      exidstr = odhdf5.getText(info,"Examples."+str(idx)+".ID."+str(idy))
      exstruct = {namedictstr: exyname, iddictstr: idy, dbkeydictstr: exidstr}
      survstr = "Examples."+str(idx)+".Survey."+str(idy)
      if odhdf5.hasAttr( info, survstr ):
        exstruct.update({locationdictstr: odhdf5.getText(info,survstr)})
      example = {extype: exstruct}
      idy += 1
    example.update({iddictstr: idx})
    surveystr = "Examples."+str(idx)+".Survey"
    if odhdf5.hasAttr( info, surveystr ):
      surveyfp = path.split( odhdf5.getText(info, surveystr ) )
      grouplbl = surveyfp[1]
      example.update({
        targetdictstr: odhdf5.getText( info, exname ),
        pathdictstr: surveyfp[0]
        })

    examples.update({grouplbl: example})
    idx += 1

  inp_sz = odhdf5.getIntValue(info,"Input.Size")
  idx = 0
  input = {}
  while idx < inp_sz:
    surveyfp = path.split( odhdf5.getText(info,"Input."+str(idx)+".Survey") )
    inp = {
      pathdictstr: surveyfp[0],
      iddictstr: idx
    }
    logsstr = "Input."+str(idx)+".Logs"
    if odhdf5.hasAttr( info, logsstr ):
      inp.update({logdictstr: odhdf5.getText(info, logsstr )})
    inpsizestr = "Input."+str(idx)+".Size"
    if odhdf5.hasAttr( info, inpsizestr ):
      idy = 0
      inpp_sz = odhdf5.getIntValue(info,inpsizestr)
      attriblist = list()
      while idy < inpp_sz:
        dsname = odhdf5.getText(info,"Input."+str(idx)+".Name."+str(idy))
        dbkey = odhdf5.getText(info,"Input."+str(idx)+".ID."+str(idy))
        attriblist.append({ namedictstr: dsname, iddictstr: idy, \
                            dbkeydictstr: dbkey })
        idy += 1
      if len(attriblist) > 0:
        inp.update({attribdictstr: attriblist} )

    input.update({surveyfp[1]: inp})
    idx += 1

  retinfo = {
    typedictstr: type,
    stepoutdictstr: stepout,
    classdictstr: True,
    interpoldictstr: odhdf5.getBoolValue(info,"Edge extrapolation"),
    exampledictstr: examples,
    inputdictstr: input
  }
  if odhdf5.hasAttr(info,'Model.Type' ):
    retinfo.update({plfdictstr: odhdf5.getText(info,'Model.Type')})
  if  odhdf5.hasAttr(info,versionstr):
    retinfo.update({versiondictstr: odhdf5.getText(info,versionstr)})
  h5file.close()

  if type == loglogtypestr:
    return getWellInfo( retinfo, filenm )
  elif type == seisclasstypestr:
    return getAttribInfo( retinfo, filenm )

  std_msg( "Unrecognized dataset type: ", type )
  raise KeyError

def getAttribInfo( info, filenm ):
  if not info[classdictstr]:
    return info

  info.update( {classesdictstr: getClassIndices(info)} )
  return info

def getWellInfo( info, filenm ):
  h5file = h5py.File( filenm, "r" )
  infods = odhdf5.getInfoDataSet( h5file )
  info[classdictstr] = odhdf5.getText(infods,'Target Value Type') == "ID"
  zstep = odhdf5.getDValue(infods,"Z step") 
  marker = (odhdf5.getText(infods,"Top marker"),
            odhdf5.getText(infods,"Bottom marker"))
  h5file.close()
  info.update({
    zstepdictstr: zstep,
    rangedictstr: marker,
  })
  return info

def addInfo( inpfile, plfnm, filenm ):
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
  outps = getOutputs( inpfile )
  nrout = len(outps)
  modeloutstr = 'Model.Output.'
  odhdf5.setAttr( dsinfoout, modeloutstr+'Size', str(nrout) )
  for idx in range(nrout):
    odhdf5.setAttr( dsinfoout, modeloutstr+str(idx)+'.Name', outps[idx] )

  h5fileout.close()

def getClassIndices( info, filternms=None ):
  ret = []
  for groupnm in info[exampledictstr]:
    if filternms==None or groupnm in filternms:
      ret.append( info[exampledictstr][groupnm][iddictstr] )
  return np.sort( ret )

def getOutputs( inpfile ):
  info = getInfo( inpfile )
  ret = list()
  type = info[typedictstr]
  isclassification = info[classdictstr]
  if isclassification:
    ret.append( classvalstr )
    if type == seisclasstypestr:
      ret.extend( getGroupNames(inpfile) )
    ret.append( confvalstr )
  else:
    if type == loglogtypestr:
      for groupnm in info[exampledictstr]:
        ret.append( info[exampledictstr][groupnm][targetdictstr] )
        break

  return ret

