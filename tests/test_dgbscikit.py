import sys
sys.path.insert(0, '..')

import dgbpy.keystr as dbk
import dgbpy.dgbscikit as dgbscikit
from dgbpy.dgbscikit import *
import dgbpy.hdf5 as dgbhdf5

from init_data import *

import pytest

all_data = lambda **kwaargs: (
    get_loglog_data(**kwaargs),
    get_loglog_classification_data(**kwaargs),
)

test_data_ids = ['loglog_regression', 'loglog_classification']

def default_pars():
    pars = dgbscikit.scikit_dict.copy()
    return pars

def get_default_model(info, params):
    return dgbscikit.getDefaultModel(info, params)

def getClusterGrpPars(info=None):
    return (lambda: getClusterParsKMeans(clustermethods[0][1], 2, 50, 50),
            lambda: getClusterParsMeanShift(clustermethods[1][1], 50),
            lambda: getClusterParsSpectral(clustermethods[2][1], 2, 50)
            )

def getLinearGrpPars(info=None):
    if not info:
        return (getLinearPars, getLogPars)
    isclassification = info[dgbkeys.classdictstr]
    if isclassification:
        return (getLogPars,)
    return (getLinearPars,)

def getEnsembleGrpPars(info=None):
    if hasXGBoost():
        return (getEnsembleParsXGDT, getEnsembleParsXGRF, getEnsembleParsRF,\
                getEnsembleParsGB, getEnsembleParsAda)
    return (getEnsembleParsRF, getEnsembleParsGB, getEnsembleParsAda)

def getNNGrpPars(info=None):
    return (getNNPars,)

def getSVMGrpPars(info=None):
    return (getSVMPars, )

def get_model_param_dict(info = None):
    cluster_pars = getClusterGrpPars()
    linear_pars = getLinearGrpPars(info)
    ensemble_pars = getEnsembleGrpPars()
    nn_pars = getNNGrpPars()
    svm_pars = getSVMGrpPars()
    return (*cluster_pars, *linear_pars, *ensemble_pars, *nn_pars, *svm_pars)


def model_init(info, params_fnc):
    params = default_pars()
    setup_param = params_fnc()
    params.update(setup_param)
    model = get_default_model(info, params)
    return model,params

def remove_model_files(filename):
    extnms = ['.onnx', '.json', '.joblib', '.h5',]
    for extnm in extnms:
        file = f'{filename}{extnm}'
        if os.path.exists(file):
            os.remove(file)

@pytest.mark.parametrize("data", all_data(), ids=test_data_ids)
def test_model_init(data):
    info = data[dbk.infodictstr]
    setup_fncs = get_model_param_dict()
    for setup_fnc in setup_fncs:
        model, _ = model_init(info, setup_fnc)
        assert model is not None, 'Model should not be of Nonetype'

@pytest.mark.parametrize("data", all_data(flatten=True), ids=test_data_ids)
def test_train_default(data):
    info = data[dbk.infodictstr]
    for setup_fnc in get_model_param_dict(info):
        model, _ = model_init(info, setup_fnc)
        model = dgbscikit.train(model, data)
        assert model is not None, 'Model should not be of Nonetype'

# @pytest.mark.parametrize("data", all_data(flatten=True), ids=test_data_ids)
# def test_saving_and_loading_model(data):
#     filenm = 'scikitmodel'
#     hdfnm = f'{filenm}.h5'
#     info = data[dbk.infodictstr]
#     _savetype = defsavetype
#     for setup_fnc in get_model_param_dict(info):
#         model, params  = model_init(info, setup_fnc)
#         model = dgbscikit.train(model, data)
        
#         isClustering = params['modelname'] == 'Clustering'
#         if isClustering:
#             _savetype = savetypes[1]

#         dgbscikit.save(model, hdfnm, _savetype)
#         model = dgbscikit.load(hdfnm)
#         assert model is not None, 'Model should not be of Nonetype'
#         remove_model_files(filenm)


    