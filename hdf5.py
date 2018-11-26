#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : Arnaud
# DATE     : November 2018
#
# tools for reading hdf5 files for NN training
#


import numpy as np
import h5py
import odpy.hdf5 as odhdf5

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

def getCubeLets( filenm, groupnm ):
  h5file = h5py.File( filenm, "r" )
  group = h5file[groupnm]
  cubelets = list()  
  for dsetnm in group:
    dset = group[dsetnm]
    # Do something with the position?
    cubelet = np.array( dset )
    cubelets.append( cubelet.squeeze() )
  h5file.close()
  return cubelets

def getAllCubeLets( filenm ):
  groupnms = getGroupNames( filenm )
  cubelets = list()
  for groupnm in groupnms:
    cubelets.append( { 'group': groupnm,
                       'data': getCubeLets(filenm,groupnm)
                     })
  return cubelets

def validInfo( info ):
  try:
    type = odhdf5.getText(info,"Type")
  except KeyError:
    print("No type found. Probably wrong type of hdf5 file")
    return False
  return True

def getBaseInfo( info, reqtype ):
  if not validInfo( info ):
    return {}

  type = odhdf5.getText(info,'Type')
  if type != reqtype:
    print( "Wrong type of training dataset. Should be: "+reqtype )
    raise KeyError

  stepout = odhdf5.getIStepInterval(info,"Stepout") 
  ex_sz = odhdf5.getIntValue(info,"Examples.Size") 
  idx = 0
  examples = list()
  while idx < ex_sz:
    namestr = "Examples."+str(idx)+".Name"
    logstr = "Examples."+str(idx)+".Log"
    if odhdf5.hasAttr( info, namestr ):
      exname = namestr
      extype = "Point-Set"
    elif odhdf5.hasAttr( info, logstr ):
      exname = logstr
      extype = "Log"
    else:
      raise KeyError
    example_sz = odhdf5.getIntValue(info,"Examples."+str(idx)+".Size")
    example_id = list()
    idy = 0
    while idy < example_sz:
      example_id.append(odhdf5.getText(info,"Examples."+str(idx)+".ID."+str(idy)))
      idy += 1
    example = {
      "name": odhdf5.getText( info, exname ),
      "type": extype,
      "id": example_id
    }
    surveystr = "Examples."+str(idx)+".Survey"
    if odhdf5.hasAttr( info, surveystr ):
      surveynm = odhdf5.getText(info, surveystr )
      example.update({'Survey': odhdf5.getText(info, surveystr )})

    examples.append( example )
    idx += 1

  inp_sz = odhdf5.getIntValue(info,"Input.Size")
  idx = 0
  input = list()
  while idx < inp_sz:
    inp = {
      "survey": odhdf5.getText(info,"Input."+str(idx)+".Survey")
    }
    logsstr = "Input."+str(idx)+".Logs"
    if odhdf5.hasAttr( info, logsstr ):
      inp.update({'Logs': odhdf5.getText(info, logsstr )})
    inpsizestr = "Input."+str(idx)+".Size"
    if odhdf5.hasAttr( info, inpsizestr ):
      inp_id = list()
      idy = 0
      inpp_sz = odhdf5.getIntValue(info,inpsizestr)
      while idy < inpp_sz:
        inp_id.append(odhdf5.getText(info,"Input."+str(idx)+".ID."+str(idy)))
        idy += 1
      inp.update({'id': inp_id})

    input.append( inp )
    idx += 1

  return {
    'type': type,
    'stepout': stepout,
    'interpolated': odhdf5.getBoolValue(info,"Edge extrapolation"),
    'examples': examples,
    'input': input
  }

def getAttribInfo( filenm ):
  h5file = h5py.File( filenm, "r" )
  infods = odhdf5.getInfoDataSet( h5file )
  baseinfo = getBaseInfo( infods, 'Seismic Classification' )
  nrsurveys = odhdf5.getIntValue( infods, 'Number of Surveys' )
  survlist = list()
  idx = 0
  while idx < nrsurveys:
    survlist.append( odhdf5.getText(infods,"Survey."+str(idx)) )
    idx += 1

  h5file.close()
  return {
    'base': baseinfo,
    'surveys': survlist
  }

def getWellInfo( filenm ):
  h5file = h5py.File( filenm, "r" )
  infods = odhdf5.getInfoDataSet( h5file )
  baseinfo = getBaseInfo( infods, 'Log-Log Prediction' )
  zstep = odhdf5.getDValue(infods,"Z step") 
  marker = (odhdf5.getText(infods,"Top marker"),
            odhdf5.getText(infods,"Bottom marker"))
  h5file.close()
  return {
    'base': baseinfo,
    'zstep': zstep,
    'range': marker,
  }
