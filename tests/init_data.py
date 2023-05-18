import sys
sys.path.insert(0, '..')
import dgbpy.keystr as dbk
import dgbpy.dgbscikit as dgbscikit
import dgbpy.hdf5 as dgbhdf5
import dgbpy.mlapply as dgbml
import dgbpy.mlio as dgbmlio
import numpy as np


def get_default_examples():
    retinfo = {
        "Dummy": {
            "target": "Dummy",
            "id": 0,
            "collection": {"Dummy": {"dbkey": "100050.1", "id": 0}},
        }
    }
    return retinfo

def get_default_multiple_examples():
    retinfo = {
        "Dummy": {
            "target": "Dummy",
            "id": 0,
            "collection": {
                "Dummy":  {"dbkey": "100050.1", "id": 0},
                "Dummy2": {"dbkey": "100050.2", "id": 1},
                "Dummy3": {"dbkey": "100050.3", "id": 3},
            },
        }
    }
    return retinfo


def get_default_input():
    retinfo = {
        "Dummy": {
            "collection": {"Dummy": {"id": 0}},
            "id": 0,
            "scale": dgbscikit.StandardScaler(),
        }
    }
    return retinfo

def get_default_multiple_input():
    retinfo = {
        "Dummy": {
            "collection": {
                "Dummy":  {"id": 0},
                "Dummy2": {"id": 1},
                "Dummy3": {"id": 2},
            },
            "id": 0,
            "scale": dgbscikit.StandardScaler(),
        }
    }
    return retinfo


def get_dataset_dict(nrpts):
    retinfo = {"Dummy": {"Dummy": list(range(nrpts))}}
    return retinfo

def get_dataset_dict_multiple(nrpts):
    retinfo = {
        "Dummy": {
            "Dummy": list(range(nrpts)),
            "Dummy2": list(range(nrpts)),
            "Dummy3": list(range(nrpts)),
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
        dbk.filedictstr: "dummy",
        dbk.estimatedsizedictstr: 1,
    }
    return retinfo

def getExampleInfos(infodict):
    learntype = infodict[dbk.learntypedictstr]
    classification = infodict[dbk.classdictstr]
    inpshape = infodict[dbk.inpshapedictstr]
    nrattribs = dgbhdf5.getNrAttribs(infodict)
    if isinstance(inpshape, int):
        if inpshape > 1:
            nrdims = 1
        else:
            nrdims = inpshape
    else:
        nrdims = len(inpshape)
    return (learntype, classification, nrdims, nrattribs)


def prepare_dataset_dict(info, nbchunks=1, seed=0, split=0.2, nbfolds=0):
    dsets = dgbmlio.getChunks(info[dbk.datasetdictstr], nbchunks)
    datasets = []
    for dset in dsets:
        if dgbhdf5.isLogInput(info) and nbfolds:
            datasets.append(
                dgbmlio.getCrossValidationIndices(
                    dset, seed=seed, valid_inputs=split, nbfolds=nbfolds
                )
            )
        else:
            datasets.append(dgbmlio.getDatasetNms(dset, validation_split=split))
    info.update({dbk.trainseldicstr: datasets, dbk.seeddictstr: seed})
    return info

def prepare_array_by_info(infos, ifold, ichunk):
    if ifold and dgbhdf5.isCrossValidation( infos ):
        datasets = infos[dbk.trainseldicstr][ichunk][dbk.foldstr+f'{ifold}']
    else:
        datasets = infos[dbk.trainseldicstr][ichunk]


def prepare_data_arr(info, split, nrpts):
    valid_nrpts = int(split * nrpts)
    inp_shape = info[dbk.inpshapedictstr]
    out_shape = info[dbk.outshapedictstr]

    nrattribs = dgbhdf5.getNrAttribs(info)
    x_train_shape = dgbhdf5.get_np_shape(inp_shape, nrpts, nrattribs)
    if isinstance(out_shape, int):
        out_shape = (out_shape,)
    if 1 not in out_shape:
        out_shape = (1, *out_shape)
    y_train_shape = (nrpts, *out_shape)

    x_valid_shape = dgbhdf5.get_np_shape(inp_shape, valid_nrpts, nrattribs)
    y_valid_shape = (valid_nrpts, *out_shape)

    if dgbhdf5.isClassification(info):
        nclasses = dgbhdf5.getNrClasses(info)
    else:
        nclasses = None
    return nclasses, (x_train_shape, y_train_shape), (x_valid_shape, y_valid_shape)

def get_seismic_imgtoimg_info():
    default = get_default_info()
    default[dbk.learntypedictstr] = dbk.seisimgtoimgtypestr
    default[dbk.inpshapedictstr] = [1, 8, 8]
    default[dbk.outshapedictstr] = [1, 8, 8]
    default[dbk.classdictstr] = True
    default[dbk.interpoldictstr] = True
    default[dbk.classesdictstr] = [1, 2, 3, 4, 5]
    return default


def get_seismic_classification_info():
    default = get_default_info()
    default[dbk.learntypedictstr] = dbk.seisclasstypestr
    default[dbk.inpshapedictstr] = [1, 8, 8]
    default[dbk.outshapedictstr] = 1
    default[dbk.classdictstr] = True
    default[dbk.interpoldictstr] = True
    default[dbk.classesdictstr] = [1, 2, 3, 4, 5]
    return default


def get_loglog_info():
    default = get_default_info()
    default[dbk.exampledictstr] = get_default_multiple_examples()
    default[dbk.inputdictstr] = get_default_multiple_input()
    default[dbk.learntypedictstr] = dbk.loglogtypestr
    default[dbk.inpshapedictstr] = 1
    default[dbk.outshapedictstr] = 1
    default[dbk.classdictstr] = False
    default[dbk.interpoldictstr] = False
    return default


def get_2d_seismic_imgtoimg_data(nrpts=16, nbchunks=1, seed=0, split=0.2, nbfolds=0):
    info = get_seismic_imgtoimg_info()
    dataset = get_dataset_dict(nrpts)
    info[dbk.datasetdictstr] = dataset
    info = prepare_dataset_dict(info, nbchunks, seed, split, nbfolds)

    nclasses, train_shape, valid_shape = prepare_data_arr(info, split, nrpts)
    x_train_shape, y_train_shape = train_shape
    x_valid_shape, y_valid_shape = valid_shape

    x_train = np.random.random(x_train_shape).astype(np.single)
    y_train = np.random.randint(nclasses, size=y_train_shape).astype(np.single)
    x_validate = np.random.random(x_valid_shape).astype(np.single)
    y_validate = np.random.randint(nclasses, size=y_valid_shape).astype(np.single)

    return {
        dbk.xtraindictstr: x_train,
        dbk.ytraindictstr: y_train,
        dbk.xvaliddictstr: x_validate,
        dbk.yvaliddictstr: y_validate,
        dbk.infodictstr: info,
    }

def get_3d_seismic_imgtoimg_data(nrpts=16, nbchunks=1, seed=0, split=0.2, nbfolds=0):
    info = get_seismic_imgtoimg_info()
    info[dbk.inpshapedictstr] = [8, 8, 8]
    info[dbk.outshapedictstr] = [8, 8, 8]
    dataset = get_dataset_dict(nrpts)
    info[dbk.datasetdictstr] = dataset
    info = prepare_dataset_dict(info, nbchunks, seed, split, nbfolds)

    nclasses, train_shape, valid_shape = prepare_data_arr(info, split, nrpts)
    x_train_shape, y_train_shape = train_shape
    x_valid_shape, y_valid_shape = valid_shape

    x_train = np.random.random(x_train_shape).astype(np.single)
    y_train = np.random.randint(nclasses, size=y_train_shape).astype(np.single)
    x_validate = np.random.random(x_valid_shape).astype(np.single)
    y_validate = np.random.randint(nclasses, size=y_valid_shape).astype(np.single)

    return {
        dbk.xtraindictstr: x_train,
        dbk.ytraindictstr: y_train,
        dbk.xvaliddictstr: x_validate,
        dbk.yvaliddictstr: y_validate,
        dbk.infodictstr: info,
    }


def get_seismic_classification_data(nrpts=16, nbchunks=1, seed=0, split=0.2, nbfolds=0):
    info = get_seismic_classification_info()
    dataset = get_dataset_dict(nrpts)
    info[dbk.datasetdictstr] = dataset
    info = prepare_dataset_dict(info, nbchunks, seed, split, nbfolds)

    nclasses, train_shape, valid_shape = prepare_data_arr(info, split, nrpts)
    x_train_shape, y_train_shape = train_shape
    x_valid_shape, y_valid_shape = valid_shape

    x_train = np.random.random(x_train_shape).astype(np.single)
    y_train = np.random.randint(nclasses, size=y_train_shape).astype(np.single)
    x_validate = np.random.random(x_valid_shape).astype(np.single)
    y_validate = np.random.randint(nclasses, size=y_valid_shape).astype(np.single)

    return {
        dbk.xtraindictstr: x_train,
        dbk.ytraindictstr: y_train,
        dbk.xvaliddictstr: x_validate,
        dbk.yvaliddictstr: y_validate,
        dbk.infodictstr: info,
    }


def get_loglog_data(nrpts=16, nbchunks=1, seed=0, split=0.2, nbfolds=0):
    info = get_loglog_info()
    dataset = get_dataset_dict_multiple(nrpts)
    info[dbk.datasetdictstr] = dataset
    info = prepare_dataset_dict(info, nbchunks, seed, split, nbfolds)

    _, train_shape, valid_shape = prepare_data_arr(info, split, nrpts)
    x_train_shape, y_train_shape = train_shape
    x_valid_shape, y_valid_shape = valid_shape

    x_train = np.random.random(x_train_shape).astype(np.single)
    y_train = np.random.random(y_train_shape).astype(np.single)
    x_validate = np.random.random(x_valid_shape).astype(np.single)
    y_validate = np.random.random(y_valid_shape).astype(np.single)

    return {
        dbk.xtraindictstr: x_train,
        dbk.ytraindictstr: y_train,
        dbk.xvaliddictstr: x_validate,
        dbk.yvaliddictstr: y_validate,
        dbk.infodictstr: info,
    }
