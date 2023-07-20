import sys
sys.path.insert(0, '..')

import os, pytest
import dgbpy.keystr as dbk
import dgbpy.dgbkeras as dgbkeras
import dgbpy.hdf5 as dgbhdf5
import dgbpy.keras_classes as kc
import keras
from init_data import *

all_data = lambda **kwargs: (
    get_2d_seismic_imgtoimg_data(**kwargs),
    get_3d_seismic_imgtoimg_data(**kwargs),
    get_seismic_classification_data(**kwargs),
    get_loglog_data(**kwargs),
    get_loglog_classification_data(**kwargs),
)

test_data_ids = ['2D_seismic_imgtoimg', '3D_seismic_imgto_img', 'seismic_classification', 'loglog_regression', 'log_classification']


def default_pars():
    pars = dgbkeras.keras_dict
    pars['epochs'] = 1
    pars['batch'] = 2
    pars[dbk.prefercpustr] = True
    pars['tofp16'] = False
    return pars


def get_default_model(info):
    learntype, classification, nrdims, nrattribs = getExampleInfos(info)
    modeltypes = dgbkeras.getModelsByType(learntype, classification, nrdims)
    return modeltypes


def get_model_arch(info, model, model_id):
    architecture = dgbkeras.getDefaultModel(info, type=model[model_id])
    return architecture

def is_model_trained(initial, current):
    for initial_layer, current_layer in zip(initial, current):
        if not np.array_equal(initial_layer, current_layer):
            return True 
    return False

def save_model(model, filename):
    dgbkeras.save(model, filename)
    return model, filename

def load_model(filename):
    return dgbkeras.load(filename, False)


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
        elif key in ['scale']:
            assert isinstance(value, str)
        elif key in ['type']:
            assert isinstance(value, bool) or value is None
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
        assert isinstance(models[i], str), 'model type should be a string'

@pytest.mark.parametrize('data', all_data(), ids=test_data_ids)
def test_default_architecture(data):
    info = data[dbk.infodictstr]
    models = get_default_model(info)
    for imodel, _ in enumerate(models):
        arch = get_model_arch(info, models, imodel)
        assert isinstance(
            arch, keras.models.Model
        ), 'architecture should be a keras Module'

def train_model(trainpars=default_pars(), data=None):
    info = data[dbk.infodictstr]
    model = get_default_model(info)
    modelarch = get_model_arch(info, model, 0)
    model = dgbkeras.train(modelarch, data, trainpars, silent=True)
    return model, info


@pytest.mark.parametrize('data', all_data(), ids=test_data_ids)
def test_train_default(data):
    info = data[dbk.infodictstr]
    models = get_default_model(info)
    for imodel, _ in enumerate(models):
        modelarch = get_model_arch(info, models, imodel)
        default_model = keras.models.clone_model(modelarch)
        trained_model = dgbkeras.train(modelarch, data, default_pars(), silent=True)
        assert isinstance(trained_model, keras.models.Model), 'model should be a keras Module'
        assert trained_model is not None, 'model should not be None'
        assert len(trained_model.get_weights()) > 0, 'model should have parameters'
        initial_state_dict = default_model.get_weights()
        current_state_dict = trained_model.get_weights()
        assert is_model_trained(
            current_state_dict, initial_state_dict
        ), 'model should have been trained'

@pytest.mark.parametrize('data', (get_2d_seismic_imgtoimg_data(),))
def test_train_with_tensorboard(data):
    pars = default_pars()
    pars['withtensorboard'] = True
    info = data[dbk.infodictstr]
    models = get_default_model(info)
    modelarch = get_model_arch(info, models, 0)
    model = dgbkeras.train(modelarch, data, pars, silent=True)

@pytest.mark.parametrize('data', all_data(), ids=test_data_ids)
def test_train_with_augmentation(data):
    pars = default_pars()
    pars['transform'] = ['flip', 'rotate', 'scale', 'noise']
    info = data[dbk.infodictstr]
    models = get_default_model(info)
    modelarch = get_model_arch(info, models, 0)
    model = dgbkeras.train(modelarch, data, pars, silent=True)
    assert isinstance(model, keras.models.Model), 'model should be a keras Module'

@pytest.mark.parametrize('data', all_data(), ids=test_data_ids)
def test_saving_and_loading_model(data):
    filename = 'kerasmodel.h5'
    trainpars = default_pars()
    model, info = train_model(trainpars=trainpars, data=data)
    save_model(model, filename)

    assert os.path.isfile(filename), 'model should have been saved'
    assert os.path.getsize(filename) > 0, 'model file should not be empty'

    loaded_model = load_model(filename)
    assert isinstance(loaded_model, keras.models.Model), 'model should be a keras Module'
    os.remove(filename)

class TCase_TrainSequence(kc.TrainingSequence):
    def set_fold(self, ichunk, ifold):
        return self.get_data(self._trainbatch)

@pytest.mark.parametrize('data', (get_loglog_data(nbfolds=2, split=1),))
def test_train_multiple_folds(data):
    pars = default_pars()
    pars['nbfold'] = 2
    info = data[dbk.infodictstr]
    models = get_default_model(info)
    modelarch = get_model_arch(info, models, 0)
    kc.TrainingSequence = TCase_TrainSequence
    model = dgbkeras.train(modelarch, data, pars, silent=True)
    assert isinstance(model, keras.models.Model), 'model should be a keras Module'

@pytest.mark.parametrize('data', (get_loglog_data(nbchunks=2, nbfolds=2, split=1),))
def test_train_multiple_chunks_and_folds(data):
    pars = default_pars()
    pars['nbchunk'] = 2
    pars['nbfold'] = 2
    info = data[dbk.infodictstr]
    models = get_default_model(info)
    modelarch = get_model_arch(info, models, 0)
    kc.TrainingSequence = TCase_TrainSequence
    model = dgbkeras.train(modelarch, data, pars, silent=True)
    assert isinstance(model, keras.models.Model), 'model should be a keras Module'

@pytest.mark.parametrize('data', all_data(nbchunks=2), ids=test_data_ids)
def test_train_multiple_chunks(data):
    pars = default_pars()
    pars['nbchunk'] = 2
    info = data[dbk.infodictstr]
    models = get_default_model(info)
    modelarch = get_model_arch(info, models, 0)
    kc.TrainingSequence = TCase_TrainSequence
    model = dgbkeras.train(modelarch, data, pars, silent=True)
    assert isinstance(model, keras.models.Model), 'model should be a keras Module'

def check_apply_res(condition, pred):
    for key, value in condition.items():
        if isinstance(value, bool) and value or isinstance(value, list) and len(value)>0:
            assert key in pred, f'{key} should be in the output result'
            assert isinstance(pred[key], np.ndarray), f'{key} should be a numpy array'
        else:
            assert key not in pred, f'{key} should not be in the output result'

def getCurrentConditions(conditions, step):
    current_conditions = {}
    for key, value in conditions.items():
        current_conditions[key] = value[step]
    return current_conditions


@pytest.mark.parametrize('data', all_data(), ids=test_data_ids)
def test_apply(data):
    info = data[dbk.infodictstr]
    models = get_default_model(info)
    modelarch = get_model_arch(info, models, 0)
    model = dgbkeras.train(modelarch, data, default_pars(), silent=True)

    inpshape = info[dbk.inpshapedictstr]
    if isinstance(inpshape, int):
        inpshape = [inpshape]
    dictinpshape = tuple( inpshape )

    samples = data[dbk.xvaliddictstr]
    if not dgbhdf5.isLogInput(info):
        nrdims = len(inpshape) - inpshape.count(1)
        if nrdims == 2:
            samples = np.squeeze(samples, axis=1)

    isclassification = info[dbk.classdictstr]

    # Get probabilities and confidence only if isLogInput
    if isclassification and dgbhdf5.isLogInput(info):
        withprobs = list(range(dgbhdf5.getNrClasses(info)))
    else:
        withprobs = []

    doprobabilities = len(withprobs) > 0

    nsteps = 4
    conditions = {
        dbk.preddictstr: [True, True, False, False],
        dbk.confdictstr: [False, True, True, False] if isclassification and dgbhdf5.isLogInput(info) else [False]*nsteps,
        dbk.probadictstr: [withprobs]*nsteps
    }
    
    for step in range(nsteps):
        withpred = conditions[dbk.preddictstr][step]
        withconfidence = conditions[dbk.confdictstr][step]
        current_conditions = getCurrentConditions(conditions, step)
        
        pred = dgbkeras.apply(model, samples, isclassification, withpred=withpred, withprobs=withprobs, withconfidence=withconfidence, doprobabilities=doprobabilities, \
                                dictinpshape=dictinpshape, scaler=None, batch_size=4)
        
        check_apply_res(current_conditions, pred)
