import sys
sys.path.insert(0, '..')

import os
import fnmatch, shutil, copy, pytest
from functools import partial
import dgbpy.keystr as dbk
import dgbpy.dgbtorch as dgbtorch
import dgbpy.hdf5 as dgbhdf5
import dgbpy.torch_classes as tc
import torch
import torch.nn as nn

from init_data import *

all_data = lambda **kwargs: (
    get_2d_seismic_imgtoimg_data(**kwargs),
    get_3d_seismic_imgtoimg_data(**kwargs),
    get_seismic_classification_data(**kwargs),
    get_loglog_data(**kwargs),
)

test_data_ids = ['2D_seismic_imgtoimg', '3D_seismic_imgto_img', 'seismic_classification', 'loglog']


def default_pars():
    pars = dgbtorch.torch_dict.copy()
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


def train_model(trainpars=default_pars(), data=None):
    info = data[dbk.infodictstr]
    model = get_default_model(info)
    modelarch = get_model_arch(info, model, 0)
    model = dgbtorch.train(modelarch, data, trainpars)
    return model, info


def is_model_trained(initial, current):
    return any(not tc.torch.equal(initial[key], current[key]) for key in initial)


def save_model(model, filename, info):
    dgbtorch.save(model, filename, info)
    return model, filename, info


def load_model(filename):
    model = dgbtorch.load(filename)
    return model


def test_training_parameters():
    pars = default_pars()
    assert isinstance(pars, dict)
    for key, value in pars.items():
        assert isinstance(key, str)
        if key in ['nbchunk', 'epochs', 'patience', 'epochdrop', 'nbfold', 'batch']:
            assert isinstance(value, int) and not isinstance(
                value, bool
            ), f'{key} must be an integer'
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


@pytest.mark.parametrize('data', all_data(), ids=test_data_ids)
def test_default_model(data):
    info = data[dbk.infodictstr]
    models = get_default_model(info)
    assert isinstance(models, list), 'model should be a list'
    assert len(models) > 0, 'model types should not be empty for dummy workflows'
    for i in range(len(models)):
        assert isinstance(models[i], str)


@pytest.mark.parametrize('data', all_data(), ids=test_data_ids)
def test_default_architecture(data):
    info = data[dbk.infodictstr]
    models = get_default_model(info)
    for imodel, _ in enumerate(models):
        arch = get_model_arch(info, models, imodel)
        assert isinstance(
            arch, dgbtorch.nn.Module
        ), 'architecture should be a nn.Module'


@pytest.mark.parametrize('data', all_data(), ids=test_data_ids)
def test_train_default(data):
    info = data[dbk.infodictstr]
    models = get_default_model(info)
    for imodel, _ in enumerate(models):
        modelarch = get_model_arch(info, models, imodel)
        default_model = copy.deepcopy(modelarch)
        trained_model = dgbtorch.train(modelarch, data, default_pars())

        assert isinstance(trained_model, nn.Module), f'model-{models[imodel]} should be a nn.Module after training'
        assert trained_model.state_dict() is not None, f'model-{models[imodel]} should have a state dict after training'

        initial_state_dict = default_model.state_dict()
        current_state_dict = trained_model.state_dict()
        assert is_model_trained(
            current_state_dict, initial_state_dict
        ), 'model should have been trained'


@pytest.mark.parametrize('data', (get_2d_seismic_imgtoimg_data(),))
def test_train_with_tensorboard(data):
    pars = default_pars()
    pars['withtensorboard'] = True
    info = data[dbk.infodictstr]
    model = get_default_model(info)
    modelarch = get_model_arch(info, model, 0)
    trained_model = dgbtorch.train(modelarch, data, pars, logdir='runs')
    assert os.path.exists('runs'), 'tensorboard file not found'
    assert isinstance(trained_model, nn.Module), 'model should be a nn.Module'
    assert any(
        fnmatch.fnmatch(filename, 'events.out.tfevents*')
        for filename in os.listdir('runs')
    ), 'tensorboard file not found'
    shutil.rmtree('runs')

@pytest.mark.parametrize('data', all_data(), ids=test_data_ids)
def test_train_with_augmentation(data):   
    get_3d_seismic_imgtoimg_data()
    pars = default_pars()
    pars['transform'] = ['Flip', 'GaussianNoise', 'Rotate', 'FlipPolarity']
    info = data[dbk.infodictstr]
    model = get_default_model(info)
    modelarch = get_model_arch(info, model, 0)
    trained_model = dgbtorch.train(modelarch, data, pars)
    assert isinstance(trained_model, nn.Module), 'model should be a nn.Module'

@pytest.mark.parametrize('data', all_data(), ids=test_data_ids)
def test_saving_and_loading_model(data):
    filename = 'torchmodel'
    trainpars = default_pars()
    model, info = train_model(data=data, trainpars=trainpars)
    save_model(model, f'{filename}.h5', info)

    assert os.path.exists(f'{filename}.h5'), 'model h5 file not found'
    assert os.path.exists(f'{filename}.onnx'), 'model onnx file not found'
    loaded_model = load_model(f'{filename}.h5')
    assert isinstance(
        loaded_model, dgbtorch.tc.OnnxModel
    ), 'loaded model should be an onnx model'
    os.remove(f'{filename}.h5')
    os.remove(f'{filename}.onnx')

class TCase_SeismicTrainDataset(tc.SeismicTrainDataset):
    def set_fold(self, ichunk, ifold):
        return self.get_data(self.imgdp, ichunk)
    
class TCase_SeismicTestDataset(tc.SeismicTestDataset):
    def set_fold(self, ichunk, ifold):
        return self.get_data(self.imgdp, ichunk)

@pytest.mark.parametrize('data', (get_loglog_data(nbfolds=2, split=1),))
def test_train_multiple_folds(data):
    pars = default_pars()
    pars['nbfold'] = 2
    info = data[dbk.infodictstr]
    model = get_default_model(info)
    modelarch = get_model_arch(info, model, 0)
    tc.SeismicTrainDataset = TCase_SeismicTrainDataset
    tc.SeismicTestDataset = TCase_SeismicTestDataset
    trained_model = dgbtorch.train(modelarch, data, pars)
    assert isinstance(trained_model, nn.Module), 'model should be a nn.Module'

@pytest.mark.parametrize('data', (get_loglog_data(nbchunks=2, nbfolds=2, split=1),))
def test_train_multiple_chunks_and_folds(data):
    pars = default_pars()
    pars['nbchunk'] = 2
    pars['nbfold'] = 2
    info = data[dbk.infodictstr]
    model = get_default_model(info)
    modelarch = get_model_arch(info, model, 0)
    tc.SeismicTrainDataset = TCase_SeismicTrainDataset
    tc.SeismicTestDataset = TCase_SeismicTestDataset
    trained_model = dgbtorch.train(modelarch, data, pars)
    assert isinstance(trained_model, nn.Module), 'model should be a nn.Module'

@pytest.mark.parametrize('data', all_data(nbchunks=2), ids=test_data_ids)
def test_train_multiple_chunks(data):
    pars = default_pars()
    pars['nbchunk'] = 2
    info = data[dbk.infodictstr]
    model = get_default_model(info)
    modelarch = get_model_arch(info, model, 0)
    trained_model = dgbtorch.train(modelarch, data, pars)
    assert isinstance(trained_model, nn.Module), 'model should be a nn.Module'
