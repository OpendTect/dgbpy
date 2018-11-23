#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : Arnaud
# DATE     : November 2018
#
# tools for reading hdf5 files for NN training
#


import collections
import numpy as np
import h5py
import odpy.hdf5 as odhdf5


def getWellInfo( filenm ):
  h5file = h5py.File( filenm, "r" )
  infods = odhdf5.getInfoDataSet( h5file )
  try:
    type = odhdf5.getText(infods,"Type")
  except KeyError:
    print("No type found. Probably wrong type of hdf5 file")
    return {}

  ex_sz = odhdf5.getIntValue(infods,"Examples.Size") 
  idx = 0
  examples = list()
  while idx < ex_sz:
    example_sz = odhdf5.getIntValue(infods,"Examples."+str(idx)+".Size")
    example_id = list()
    idy = 0
    while idy < example_sz:
      example_id.append(odhdf5.getText(infods,"Examples."+str(idx)+".ID."+str(idy)))
      idy += 1
    example = {
      "name": odhdf5.getText(infods,"Examples."+str(idx)+".Log"),
      "id": example_id,
      "survey": odhdf5.getText(infods,"Examples."+str(idx)+".Survey")
    }
    examples.append( example )
    idx += 1

  inp_sz = odhdf5.getIntValue(infods,"Input.Size")
  idx = 0
  input = list()
  while idx < inp_sz:
    inp = collections.OrderedDict({
      "name": odhdf5.getText(infods,"Input."+str(idx)+".Logs"),
      "survey": odhdf5.getText(infods,"Input."+str(idx)+".Survey")
    })
    input.append( inp )
    idx += 1

  zstep = odhdf5.getDValue(infods,"Z step") 
  stepout = odhdf5.getIntValue(infods,"Stepout") 
  marker = (odhdf5.getText(infods,"Top marker"),
            odhdf5.getText(infods,"Bottom marker"))
  isinterpol = odhdf5.getBoolValue(infods,"Edge extrapolation")
  h5file.close()
  return collections.OrderedDict({
    'examples': examples,
    'input': input,
    'zstep': zstep,
    'stepout': stepout,
    'marker': marker,
    'interpolated': isinterpol
  })
