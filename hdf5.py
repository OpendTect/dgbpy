#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : Arnaud
# DATE     : November 2018
#
# tools for reading hdf5 files for NN training
#

from os import path
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

def get_np_shape( step, nrpts=None ):
  ret = ()
  if nrpts != None:
    ret += (nrpts,)
  if isinstance( step, int ):
    ret += ( step*2+1, )
    return ret
  for i in step:
    ret += (i*2+1,)
  return ret

def getCubeLets( filenm, infos, groupnm, decim ):
  nrattribs = 0
  try:
    nrattribs = len( infos['input'][groupnm]['Logs'] )
  except KeyError:
    nrattribs = 1
  stepout = infos['stepout']
  isclass = infos['classification']
  if decim != None:
    if decim < 0 or decim > 100:
      print( "Decimation percentage not within [0,100]" )
      raise ValueError
  h5file = h5py.File( filenm, "r" )
  group = h5file[groupnm]
  dsetnms = list(group.keys())
  nrpts = len(dsetnms)
  if decim != None:
    np.random.shuffle( dsetnms )
    nrpts = int(nrpts*(decim/100))
    if nrpts < 1:
      return {}
    del dsetnms[nrpts:]
  shape = None
  if nrattribs==1:
    shape = get_np_shape(stepout,nrpts)
  else:
    if stepout > 0:
      shape = ( nrpts, nrattribs, stepout*2+1 )
    else:
      shape = ( nrpts, nrattribs )

  cubelets = np.empty( shape, np.float32 )
  outdtype = np.float32
  if isclass:
    outdtype = np.uint8
  output = np.empty( nrpts, outdtype )
  idx = 0
  for dsetnm in dsetnms:
    dset = group[dsetnm]
    cubelets[idx] = np.array( dset ).squeeze()
    if isclass :
      output[idx] = odhdf5.getIntValue( dset, 'Value' )
    else:
      output[idx] = odhdf5.getDValue( dset, 'Value' )
    idx += 1

  h5file.close()
  ret = {
    'x': cubelets,
    'y': output
  }
  return ret

def getAllCubeLets( filenm, decim=None ):
  infos = getInfo( filenm )
  groupnms = getGroupNames( filenm )
  cubelets = list()
  for groupnm in groupnms:
    cubelets.append( getCubeLets(filenm,infos,groupnm,decim) )
  totsz = 0
  allx = list()
  ally = list()
  for cubelet in cubelets:
    totsz += len(cubelet['x'])
    allx.append( cubelet['x'] )
    ally.append( cubelet['y'] )
  return {
    'x': np.concatenate( allx ),
    'y': np.concatenate( ally )
  }

def validInfo( info ):
  try:
    type = odhdf5.getText(info,"Type")
  except KeyError:
    print("No type found. Probably wrong type of hdf5 file")
    return False
  return True

def getInfo( filenm ):
  h5file = h5py.File( filenm, "r" )
  info = odhdf5.getInfoDataSet( h5file )
  if not validInfo( info ):
    h5file.close()
    return {}

  type = odhdf5.getText(info,'Type')
  stepout = odhdf5.getIStepInterval(info,"Stepout") 
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
      exstruct = {'name': exyname, 'id': idy, 'dbkey': exidstr}
      survstr = "Examples."+str(idx)+".Survey."+str(idy)
      if odhdf5.hasAttr( info, survstr ):
        exstruct.update({'location': odhdf5.getText(info,survstr)})
      example = {extype: exstruct}
      idy += 1
    example.update({'id': idx})
    surveystr = "Examples."+str(idx)+".Survey"
    if odhdf5.hasAttr( info, surveystr ):
      surveyfp = path.split( odhdf5.getText(info, surveystr ) )
      grouplbl = surveyfp[1]
      example.update({
        'target': odhdf5.getText( info, exname ),
        'path': surveyfp[0]
        })

    examples.update({grouplbl: example})
    idx += 1

  inp_sz = odhdf5.getIntValue(info,"Input.Size")
  idx = 0
  input = {}
  while idx < inp_sz:
    surveyfp = path.split( odhdf5.getText(info,"Input."+str(idx)+".Survey") )
    inp = {
      'path': surveyfp[0],
      'id': idx
    }
    logsstr = "Input."+str(idx)+".Logs"
    if odhdf5.hasAttr( info, logsstr ):
      inp.update({'Logs': odhdf5.getText(info, logsstr )})
    inpsizestr = "Input."+str(idx)+".Size"
    if odhdf5.hasAttr( info, inpsizestr ):
      idy = 0
      inpp_sz = odhdf5.getIntValue(info,inpsizestr)
      while idy < inpp_sz:
        dsname = odhdf5.getText(info,"Input."+str(idx)+".Name."+str(idy))
        dbkey = odhdf5.getText(info,"Input."+str(idx)+".ID."+str(idy))
        inpstruct = { 'name': dsname, 'id': idy, 'dbkey': dbkey } 
        inp.update({'Attributes': inpstruct })
        idy += 1

    input.update({surveyfp[1]: inp})
    idx += 1

  info = {
    'type': type,
    'stepout': stepout,
    'classification': True,
    'interpolated': odhdf5.getBoolValue(info,"Edge extrapolation"),
    'examples': examples,
    'input': input
  }
  h5file.close()

  if type == 'Log-Log Prediction':
    return getWellInfo( info, filenm )
  elif type == 'Seismic Classification':
    return info

  print( "Unrecognized dataset type: ", type )
  raise KeyError

def getWellInfo( info, filenm ):
  h5file = h5py.File( filenm, "r" )
  infods = odhdf5.getInfoDataSet( h5file )
  info['classification'] = odhdf5.getText(infods,'Target Value Type') == "ID"
  zstep = odhdf5.getDValue(infods,"Z step") 
  marker = (odhdf5.getText(infods,"Top marker"),
            odhdf5.getText(infods,"Bottom marker"))
  h5file.close()
  info.update({
    'zstep': zstep,
    'range': marker,
  })
  return info
