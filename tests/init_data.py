
import dgbpy.keystr as dbk
import dgbpy.dgbscikit as dgbscikit
import  dgbpy.hdf5 as dgbhdf5
import dgbpy.mlapply as dgbml
import dgbpy.mlio as dgbmlio
import numpy as np


def get_default_examples():
  retinfo = {
    'Dummy': {
                'target': 'Dummy',
                'id': 0,
                'collection': {'Dummy': {'dbkey': '100050.1', 'id': 0}
              }
    }
  }
  return retinfo

def get_default_input():
  retinfo = {
    'Dummy': {
                'collection': {'Dummy': {'id': 0}},
                'id': 0,
                'scale': dgbscikit.StandardScaler()
              }
  }
  return retinfo

def get_dataset_dict(nrpts):
  retinfo = {
    'Dummy': {
                'Dummy': list(range(nrpts))
              }
  }
  return retinfo

def get_default_info():
  retinfo = {
    dbk.learntypedictstr: dbk.loglogtypestr,
    dbk.segmentdictstr: False,
    dbk.inpshapedictstr: 1,
    dbk.outshapedictstr: 1,
    dbk.classdictstr: False,
    dbk.interpoldictstr: False,
    dbk.exampledictstr: get_default_examples(),
    dbk.inputdictstr: get_default_input(),
    dbk.filedictstr: 'dummy',
    dbk.estimatedsizedictstr: 1
  }
  return retinfo

def getExampleInfos( infodict ):
    learntype = infodict[dbk.learntypedictstr]
    classification = infodict[dbk.classdictstr]
    inpshape = infodict[dbk.inpshapedictstr]
    nrattribs = dgbhdf5.getNrAttribs(infodict)
    if isinstance(inpshape,int):
        if inpshape > 1: nrdims = 1
        else: nrdims = inpshape
    else: nrdims = len(inpshape)
    return (learntype,classification,nrdims,nrattribs)

def prepare_dataset(info, nbchunks=1, seed=0, split=.2, nbfolds=0):
  dsets = dgbmlio.getChunks(info[dbk.datasetdictstr], nbchunks)
  datasets = []
  for dset in dsets:
      if dgbhdf5.isLogInput(info) and nbfolds:
          datasets.append( dgbmlio.getCrossValidationIndices(dset,seed=seed,valid_inputs=split,nbfolds=nbfolds) )
      else:
          datasets.append( dgbmlio.getDatasetNms(dset, validation_split=split) )
  info.update({dbk.trainseldicstr: datasets, dbk.seeddictstr: seed})
  return info
   
def get_seismic_imgtoimg_info():
  default = get_default_info()
  default[dbk.learntypedictstr] = dbk.seisimgtoimgtypestr
  default[dbk.inpshapedictstr] = [1, 8, 8]
  default[dbk.outshapedictstr] = [1, 8, 8]
  default[dbk.classdictstr] = True
  default[dbk.interpoldictstr] = True
  default[dbk.classesdictstr] = [1, 2, 3, 4, 5]
  return default

def get_seismic_imgtoimg_data(nrpts=10, nbchunks=1, seed=0, split=.2, nbfolds=0):
  valid_nrpts = int(split*nrpts)

  info = get_seismic_imgtoimg_info()

  dataset = get_dataset_dict(nrpts)
  info[dbk.datasetdictstr] = dataset
  
  info = prepare_dataset(info, nbchunks, seed, split, nbfolds)

  inp_shape = info[dbk.inpshapedictstr]
  out_shape = info[dbk.outshapedictstr]

  nrattribs = dgbhdf5.getNrAttribs(info)
  x_train_shape = dgbhdf5.get_np_shape(inp_shape, nrpts, nrattribs)
  y_train_shape = (nrpts, *out_shape)

  x_valid_shape = dgbhdf5.get_np_shape(inp_shape, valid_nrpts, nrattribs)
  y_valid_shape = (valid_nrpts, *out_shape)

  nclasses = dgbhdf5.getNrClasses(info)
  x_train = np.random.random(x_train_shape).astype(np.single)
  y_train = np.random.randint(nclasses, size=y_train_shape).astype(np.single)
  x_validate = np.random.random(x_valid_shape).astype(np.single)
  y_validate = np.random.randint(nclasses, size=y_valid_shape).astype(np.single)

  return {
    dbk.xtraindictstr: x_train,
    dbk.ytraindictstr: y_train,
    dbk.xvaliddictstr: x_validate,
    dbk.yvaliddictstr: y_validate,
    dbk.infodictstr: info
  }




