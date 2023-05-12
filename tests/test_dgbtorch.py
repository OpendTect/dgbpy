import dgbpy.keystr as dbk
import dgbpy.dgbtorch as dgbtorch
import dgbpy.hdf5 as dgbhdf5
import dgbpy.torch_classes as tc

from init_data import *
import pytest

def default_pars():
    pars = dgbtorch.torch_dict
    pars['epochs'] = 1
    pars['batch'] = 2
    pars[dbk.prefercpustr] = True
    pars['tofp16'] = False
    return pars

def get_default_model(info):
    learntype, classification, nrdims, nrattribs = getExampleInfos(info)
    modeltype = dgbtorch.getModelsByType(learntype, classification, nrdims)
    return modeltype

def get_model_arch(info, model, model_id):
    architecture = dgbtorch.getDefaultModel(info, type=model[model_id])
    return architecture

def test_training_parameters():
    pars = default_pars()
    assert isinstance(pars, dict)
    for key, value in pars.items():
        assert isinstance(key, str)
        if key in ['nbchunk', 'epochs', 'patience', 'epochdrop', 'nbfold', 'batch']:
            assert isinstance(value, int) and not isinstance(value, bool), f'{key} must be an integer'
        elif key in ['split', 'learnrate']:
            assert isinstance(value, (float, int))
        elif key in ['criterion']:
            assert isinstance(value, dgbtorch.nn.modules.loss._Loss)
        elif key in ['scale']:
            assert isinstance(value, str)
        elif key in ['type']:
            assert isinstance(value, str) or value is None
        elif key in [dbk.prefercpustr, 'withtensorboard', 'tofp16']:
            assert isinstance(value, bool)
        elif key in ['transform']:
            assert isinstance(value, list)

def test_default_architecture():
    info = get_seismic_imgtoimg_data()[dbk.infodictstr]
    models = get_default_model(info)
    for imodel, _ in enumerate(models):
        arch = get_model_arch(info, models, imodel)
        assert isinstance(arch, dgbtorch.nn.Module), 'architecture should be a nn.Module'

def test_default_model():   
    info = get_seismic_imgtoimg_data()[dbk.infodictstr]
    model = get_default_model(info)
    assert isinstance(model, list), 'model should be a list'
    assert len(model) > 0, 'model types should not be empty for dummy workflows'
    for i in range(len(model)):
        assert isinstance(model[i], str)

def train_model(trainpars=default_pars()):
    data = get_seismic_imgtoimg_data()
    info = data[dbk.infodictstr]
    model = get_default_model(info)
    modelarch = get_model_arch(info, model, 0)
    model = dgbtorch.train(modelarch, data,trainpars)
    return model

def test_train_default():
    model = train_model()
    


