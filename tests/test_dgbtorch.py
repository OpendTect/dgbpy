import sys
sys.path.insert(0, '..')

import os
import fnmatch, shutil, copy, pytest
from functools import partial
import dgbpy.keystr as dbk
import dgbpy.dgbtorch as dgbtorch
from dgbpy.dgbtorch import SaveType
import dgbpy.hdf5 as dgbhdf5
if not dgbtorch.hasTorch():
    pytest.skip(allow_module_level=True)
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
    pars['nbfold'] = 1
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

def check_apply_res(condition, pred):
    for key, value in condition.items():
        if isinstance(value, bool) and value or isinstance(value, list) and len(value)>0:
            assert key in pred, f'{key} should be in the output result'
            assert isinstance(pred[key], np.ndarray), f'{key} should be a numpy array'
        else:
            assert key not in pred, f'{key} should not be in the output result'

def save_model(model, filename, info, params):
    dgbtorch.save(model, filename, info, params)
    return model, filename, info


def load_model(filename, info=None):
    model = dgbtorch.load(filename, info)
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
            assert isinstance(value, str) or value is None
        elif key in ['scale']:
            assert isinstance(value, str)
        elif key in ['type']:
            assert isinstance(value, str) or value is None
        elif key in [dbk.prefercpustr, 'withtensorboard', 'tofp16']:
            assert isinstance(value, bool)
        elif key in ['transform']:
            assert isinstance(value, list)

@pytest.mark.parametrize('data', (get_2d_seismic_imgtoimg_data(),))
def test_criterion_with_data(data):

    # Test for None paramter and pytorch loss class names
    test_criterion_params = {None: nn.CrossEntropyLoss, 'MSELoss': nn.MSELoss, 'CrossEntropyLoss': nn.CrossEntropyLoss}
    for params, criterion_class in test_criterion_params.items():
        pars = default_pars()
        pars['criterion'] = params
        info = data[dbk.infodictstr]
        criterion = dgbtorch.get_criterion(info, pars)
        assert isinstance(criterion, criterion_class), f'criterion should be a {criterion.__name__}'

    # Test for custom loss classes in dgbtorch
    class DummyCriterion(nn.Module): pass
    dgbtorch.DummyCriterion = DummyCriterion
    pars['criterion'] = 'DummyCriterion'
    criterion = dgbtorch.get_criterion(info, pars)
    assert isinstance(criterion, DummyCriterion), f'criterion should be a DummyCriterion'

    # Test for unsupported loss classes
    pars['criterion'] = 'MSE'
    with pytest.raises(ValueError): dgbtorch.get_criterion(info, pars)

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
            arch, nn.Module
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
    pars = default_pars()
    pars['transform'] = ['Flip', 'GaussianNoise', 'Rotate', 'FlipPolarity']
    info = data[dbk.infodictstr]
    model = get_default_model(info)
    modelarch = get_model_arch(info, model, 0)
    trained_model = dgbtorch.train(modelarch, data, pars)
    assert isinstance(trained_model, nn.Module), 'model should be a nn.Module'

@pytest.mark.parametrize('data', all_data(), ids=test_data_ids)
def test_saving_and_loading_model(data):
    trainpars = default_pars()
    savetypes = list(SaveType)
    info = data[dbk.infodictstr]
    filename = 'torchmodel'

    models = get_default_model(info)
    for imodel, _ in enumerate(models):
        modelarch = get_model_arch(info, models, imodel)
        modelarch = copy.deepcopy(modelarch)
        model = dgbtorch.train(modelarch, data, default_pars())
        for savetype in savetypes:            
            trainpars['savetype'] = savetype.value
            save_model(model, f'{filename}.h5', info, trainpars)
            assert os.path.exists(f'{filename}.h5'), 'model h5 file not found'

            if savetype == SaveType.Onnx:
                assert os.path.exists(f'{filename}.onnx'), 'model onnx file not found'
                loaded_model = load_model(f'{filename}.h5', info)
                assert isinstance(
                    loaded_model, dgbtorch.tc.OnnxTorchModel
                ), 'loaded model should be an onnx model'
                os.remove(f'{filename}.onnx')

            elif savetype == SaveType.TorchScript:
                assert os.path.exists(f'{filename}.pth'), 'model pth file not found'
                loaded_model = load_model(f'{filename}.h5', info)
                assert isinstance(
                    loaded_model, torch.jit.RecursiveScriptModule
                ), 'loaded model should be a pytorch script model'
                os.remove(f'{filename}.pth')

            elif savetype == SaveType.Joblib:
                assert os.path.exists(f'{filename}.joblib'), 'model joblib file not found'
                loaded_model = load_model(f'{filename}.h5', info)
                assert isinstance(
                    loaded_model, nn.Module
                ), f'loaded joblib model{model.__class__.__name__} should be a torch Module'
                os.remove(f'{filename}.joblib')

            elif savetype == SaveType.Pickle:
                loaded_model = load_model(f'{filename}.h5', info)
                assert isinstance(
                    loaded_model, nn.Module
                ), 'loaded pickle model should be a torch Module'

            os.remove(f'{filename}.h5')

class TCase_TrainDataset(tc.TrainDatasetClass):
    def set_fold(self, ichunk, ifold):
        return self.get_data(self.imgdp, ichunk)
    
class TCase_TestDataset(tc.TestDatasetClass):
    def set_fold(self, ichunk, ifold):
        return self.get_data(self.imgdp, ichunk)

@pytest.mark.parametrize('data', (get_loglog_data(nbfolds=2, split=1),))
def test_train_multiple_folds(data):
    pars = default_pars()
    pars['nbfold'] = 2
    info = data[dbk.infodictstr]
    model = get_default_model(info)
    modelarch = get_model_arch(info, model, 0)
    tc.TrainDatasetClass = TCase_TrainDataset
    tc.TestDatasetClass = TCase_TestDataset
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
    tc.TrainDatasetClass = TCase_TrainDataset
    tc.TestDatasetClass = TCase_TestDataset
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

@pytest.mark.parametrize('data', all_data(), ids=test_data_ids)
def test_apply_result_dictionary(data):
    pars = default_pars()
    info = data[dbk.infodictstr]
    model = get_default_model(info)
    modelarch = get_model_arch(info, model, 0)
    model = dgbtorch.train(modelarch, data, pars, silent=True)

    filename = 'torchmodel'
    save_model(model, f'{filename}.h5', info, pars)
    trained_model = load_model(f'{filename}.h5')	

    samples = data[dbk.xvaliddictstr]
    isclassification = info[dbk.classdictstr]
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
        pred = dgbtorch.apply(trained_model, info, samples, None, isclassification, withpred, withprobs, withconfidence, doprobabilities)
        check_apply_res(current_conditions, pred)

    os.remove(f'{filename}.h5')
    os.remove(f'{filename}.onnx')


@pytest.mark.parametrize('data',
                         (get_2d_seismic_imgtoimg_data(nrclasses=5), get_3d_seismic_imgtoimg_data(nrclasses=5)),
                         ids=['2D_seismic_imgtoimg', '3D_seismic_imgto_img'])
def test_apply_result____img2img_classes_gt_2_should_return_class_integers(data):
    pars = default_pars()
    info = data[dbk.infodictstr]
    model = get_default_model(info)
    modelarch = get_model_arch(info, model, 0)
    model = dgbtorch.train(modelarch, data, pars, silent=True)

    filename = 'torchmodel'
    save_model(model, f'{filename}.h5', info, pars)
    trained_model = load_model(f'{filename}.h5')

    samples = data[dbk.xvaliddictstr]
    isclassification = info[dbk.classdictstr]
    withpred = True
    withprobs = []
    withconfidence = False
    doprobabilities = len(withprobs) > 0

    pred = dgbtorch.apply(trained_model, info, samples, None, isclassification, withpred, withprobs, withconfidence, doprobabilities)

    assert dbk.preddictstr in pred, 'prediction should be in the output result'
    prediction = pred[dbk.preddictstr]

    # Reshape array to fit the target shape
    if getNrDims(info[dbk.inpshapedictstr]) < 3:
        prediction = prediction[:, np.newaxis, np.newaxis, ...]
    else:
        prediction = prediction[:, np.newaxis, ...]

    # Ensure the prediction shape is the same as the target shape
    yvalid = data[dbk.yvaliddictstr]
    assert prediction.shape == yvalid.shape, 'prediction shape should be the same as the target shape'

    # Ensure the prediction uses argmax to get the class number
    assert np.issubdtype(prediction.dtype, np.integer), 'prediction should be an integer array representing each class'

    os.remove(f'{filename}.h5')
    os.remove(f'{filename}.onnx')

@pytest.mark.parametrize('data',
                         (get_2d_seismic_imgtoimg_data(nrclasses=2), get_3d_seismic_imgtoimg_data(nrclasses=2)),
                         ids=['2D_seismic_imgtoimg', '3D_seismic_imgto_img'])
def test_apply_result____img2img_classes_is_2_should_return_continous_values(data):
    pars = default_pars()
    info = data[dbk.infodictstr]
    model = get_default_model(info)
    modelarch = get_model_arch(info, model, 0)
    model = dgbtorch.train(modelarch, data, pars, silent=True)

    filename = 'torchmodel'
    save_model(model, f'{filename}.h5', info, pars)
    trained_model = load_model(f'{filename}.h5')

    samples = data[dbk.xvaliddictstr]
    isclassification = info[dbk.classdictstr]
    withpred = True
    withprobs = []
    withconfidence = False
    doprobabilities = len(withprobs) > 0

    pred = dgbtorch.apply(trained_model, info, samples, None, isclassification, withpred, withprobs, withconfidence, doprobabilities)

    assert dbk.preddictstr in pred, 'prediction should be in the output result'
    prediction = pred[dbk.preddictstr]

    # Reshape array to fit the target shape
    if getNrDims(info[dbk.inpshapedictstr]) < 3:
        prediction = prediction[:, np.newaxis, np.newaxis, ...]
    else:
        prediction = prediction[:, np.newaxis, ...]

    # Ensure the prediction shape is the same as the target shape
    yvalid = data[dbk.yvaliddictstr]
    assert prediction.shape == yvalid.shape, 'prediction shape should be the same as the target shape'

    # Ensure the prediction is a continuous value
    assert np.issubdtype(prediction.dtype, np.floating), 'prediction should be a continuous value'

    os.remove(f'{filename}.h5')
    os.remove(f'{filename}.onnx')


@pytest.mark.parametrize('data',
                         (get_2d_seismic_imgtoimg_data(nrclasses=5, nr_inattr=2), get_3d_seismic_imgtoimg_data(nrclasses=5, nr_inattr=2)),
                         ids=['2D_seismic_imgtoimg', '3D_seismic_imgto_img'])
def test_train_and_apply_multiple_in_attributes(data):
    pars = default_pars()
    info = data[dbk.infodictstr]
    model = get_default_model(info)
    modelarch = get_model_arch(info, model, 0)
    model = dgbtorch.train(modelarch, data, pars, silent=True)

    print(data[dbk.xtraindictstr].shape, data[dbk.ytraindictstr].shape, data[dbk.xvaliddictstr].shape, data[dbk.yvaliddictstr].shape)

    filename = 'torchmodel'
    save_model(model, f'{filename}.h5', info, pars)
    trained_model = load_model(f'{filename}.h5')

    samples = data[dbk.xvaliddictstr]
    isclassification = info[dbk.classdictstr]
    withpred = True
    withprobs = []
    withconfidence = False
    doprobabilities = len(withprobs) > 0

    pred = dgbtorch.apply(trained_model, info, samples, None, isclassification, withpred, withprobs, withconfidence, doprobabilities)

    assert dbk.preddictstr in pred, 'prediction should be in the output result'
    prediction = pred[dbk.preddictstr]
    prediction = prediction[:, np.newaxis,...]

    # Ensure the prediction shape is the same as the target shape
    yvalid = data[dbk.yvaliddictstr]
    if getNrDims(info[dbk.inpshapedictstr]) < 3: 
        prediction = prediction[:, :, np.newaxis,...] # Because torch apply will return (n,c,x,y) for 2D instead of (n,c,x,y,z)  
    assert prediction.shape == yvalid.shape, 'prediction shape should be the same as the target shape'


@pytest.mark.parametrize('data',
                            (get_2d_seismic_imgtoimg_data(nrclasses=False, nr_outattr=2), get_3d_seismic_imgtoimg_data(nrclasses=False, nr_outattr=2)),
                            ids=['2D_seismic_imgtoimg_regression', '3D_seismic_imgto_img_regression'])
def test_train_and_apply_multiple_out_attributes(data):
    pars = default_pars()
    info = data[dbk.infodictstr]
    model = get_default_model(info)
    modelarch = get_model_arch(info, model, 0)

    model = dgbtorch.train(modelarch, data, pars, silent=True)

    filename = 'torchmodel'
    save_model(model, f'{filename}.h5', info, pars)
    trained_model = load_model(f'{filename}.h5')

    samples = data[dbk.xvaliddictstr]
    isclassification = info[dbk.classdictstr]
    withpred = True
    withprobs = []
    withconfidence = False
    doprobabilities = len(withprobs) > 0

    pred = dgbtorch.apply(trained_model, info, samples, None, isclassification, withpred, withprobs, withconfidence, doprobabilities)

    assert dbk.preddictstr in pred, 'prediction should be in the output result'
    prediction = pred[dbk.preddictstr]

    yvalid = data[dbk.yvaliddictstr]
    if getNrDims(info[dbk.inpshapedictstr]) < 3: 
        prediction = prediction[:, :, np.newaxis,...] # Because torch apply will return (n,c,x,y) for 2D instead of (n,c,x,y,z)        
    assert prediction.shape == yvalid.shape, 'prediction shape should be the same as the target shape'

@pytest.mark.parametrize('data',
                            (get_2d_seismic_imgtoimg_data(nrclasses=False, nr_inattr=2, nr_outattr=2),
                             get_3d_seismic_imgtoimg_data(nrclasses=False, nr_inattr=2, nr_outattr=2)),
                            ids=['2D_seismic_imgtoimg_regression', '3D_seismic_imgto_img_regression'])
def test_train_and_apply_multiple_in_and_out_attributes(data):
    pars = default_pars()
    info = data[dbk.infodictstr]
    model = get_default_model(info)
    modelarch = get_model_arch(info, model, 0)
    model = dgbtorch.train(modelarch, data, pars, silent=True)

    filename = 'torchmodel'
    save_model(model, f'{filename}.h5', info, pars)
    trained_model = load_model(f'{filename}.h5')

    samples = data[dbk.xvaliddictstr]
    isclassification = info[dbk.classdictstr]
    withpred = True
    withprobs = []
    withconfidence = False
    doprobabilities = len(withprobs) > 0

    pred = dgbtorch.apply(trained_model, info, samples, None, isclassification, withpred, withprobs, withconfidence, doprobabilities)

    assert dbk.preddictstr in pred, 'prediction should be in the output result'
    prediction = pred[dbk.preddictstr]

    yvalid = data[dbk.yvaliddictstr]
    if getNrDims(info[dbk.inpshapedictstr]) < 3:
        prediction = prediction[:, :, np.newaxis,...]
    assert prediction.shape == yvalid.shape, 'prediction shape should be the same as the target shape'


