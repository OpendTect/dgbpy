import sys
sys.path.insert(0, '..')

import os
import fnmatch, pytest
import dgbpy.keystr as dbk
import dgbpy.mlapply as dgbml
import dgbpy.dgbkeras as dgbkeras
from init_data import *
if dgbkeras.hasKeras():
    from test_dgkeras import default_pars as keras_params
from test_dgbscikit import default_pars as scikit_params
from test_dgbtorch import default_pars as torch_params

storageFolder = 'tests/examples'
@pytest.fixture(autouse=True)
def skip_if_folder_missing(request):
    storage_folder_name = request.node.get_closest_marker('skip_if_folder_missing')
    if storage_folder_name:
        if not os.path.exists(storage_folder_name.args[0]):
            pytest.skip("Folder missing: %s" % storage_folder_name.args[0])

def get_example_files():
    examplefilenm = []
    if not os.path.exists(storageFolder):
        return examplefilenm
    for file in os.listdir(storageFolder):
        if fnmatch.fnmatch(file, '*.h5'):
            examplefilenm.append(os.path.join(storageFolder, file))
    return examplefilenm

def get_filenm_from_path(filepath):
    filename = os.path.basename(filepath)
    name, _ = os.path.splitext(filename)
    return name


test_data_ids = []
examples = get_example_files()


def keras_test_cases(examplefilenm=''):
    params = keras_params()
    examplefilenm = get_filenm_from_path(examplefilenm)
    return {
        'platform': dbk.kerasplfnm,
        'type': dgbml.TrainType.New,
        'params': params,
        'outnm': f'keras_test_{examplefilenm}.h5',
        'logdir': None,
        'clearlogs': False,
        'modelin': None,
    }

@pytest.mark.skip_if_folder_missing(storageFolder)
@pytest.mark.parametrize('examplefilenm', [examples])
def test_doTrain_invalid_platform(examplefilenm, capsys):
    kwargs = {
        'platform': 'invalid_platform',
        'type': dgbml.TrainType.New,
        'params': {},
        'outnm': 'invalid_platform_test.h5',
        'logdir': None,
        'clearlogs': False,
        'modelin': None,
    }
    with pytest.raises(AttributeError):
        dgbml.doTrain(examplefilenm, **kwargs)
        assert dgbml.doTrain(os.path.join(storageFolder, 'invalid_platform_test.h5'), **kwargs) == False
        captured = capsys.readouterr()
        assert 'Unsupported machine learning platform' in captured.out

@pytest.mark.parametrize('examplefilenm', examples)
def test_doTrain_keras_new_trainingtype(examplefilenm):
    kwargs = keras_test_cases()
    assert dgbml.doTrain(examplefilenm, **kwargs) == True

@pytest.mark.parametrize('examplefilenm', examples)
def test_doTrain_keras_resume_trainingtype(examplefilenm):
    kwargs = keras_test_cases(examplefilenm)
    if not os.path.exists(kwargs['outnm']):
        dgbml.doTrain(examplefilenm, **kwargs)
    kwargs['modelin'] = kwargs['outnm']
    kwargs['type'] = dgbml.TrainType.Resume
    assert dgbml.doTrain(examplefilenm, **kwargs) == True

@pytest.mark.parametrize('examplefilenm', examples)
def test_doTrain_keras_transfer_trainingtype(examplefilenm):
    kwargs = keras_test_cases(examplefilenm)
    if not os.path.exists(kwargs['outnm']):
        dgbml.doTrain(examplefilenm, **kwargs) # Create pretrained model
    kwargs['modelin'] = kwargs['outnm']
    kwargs['type'] = dgbml.TrainType.Transfer
    assert dgbml.doTrain(examplefilenm, **kwargs) == True

def torch_test_cases(examplefilenm):
    params = torch_params()
    params['nbfold'] = None
    params['nbchunk'] = 1

    # Use larger batch size for log inputs
    if dgbhdf5.isLogInput(dgbhdf5.getInfo(examplefilenm, True)): params['batch'] = 64

    examplefilenm = get_filenm_from_path(examplefilenm)
    outnm = f'torch_test_{examplefilenm}.h5'
    return {
        'platform': dbk.torchplfnm,
        'type': None,
        'params': params,
        'outnm': outnm,
        'logdir': None,
        'clearlogs': False,
        'modelin': None,
    }

models = [] # Use these models for continue training
def get_pretrained_modelfilenm(examplefilenm):
    examplefilenm = get_filenm_from_path(examplefilenm)
    for model in models:
        if examplefilenm in model:
            return model
    return None

    

@pytest.mark.parametrize('examplefilenm', examples)
def test_doTrain_torch_new_trainingtype(examplefilenm):
    kwargs = torch_test_cases(examplefilenm)
    assert dgbml.doTrain(examplefilenm, **kwargs) == True
    models.append(kwargs['outnm'])

@pytest.mark.parametrize('examplefilenm', examples)
def test_doTrain_torch_resume_trainingtype(examplefilenm):
    kwargs = torch_test_cases(examplefilenm)

    pretrained_model_filenm = get_pretrained_modelfilenm(examplefilenm)
    if not pretrained_model_filenm or not os.path.exists(pretrained_model_filenm):
        dgbml.doTrain(examplefilenm, **kwargs) # Create a new pretrained model
        pretrained_model_filenm = kwargs['outnm']

    kwargs['type'] = dgbml.TrainType.Resume
    kwargs['modelin'] = pretrained_model_filenm
    kwargs['outnm'] = f'torch_test_{get_filenm_from_path(examplefilenm)}_resume.h5' 
    assert dgbml.doTrain(examplefilenm, **kwargs) == True    

@pytest.mark.parametrize('examplefilenm', examples)
def test_doTrain_torch_transfer_trainingtype(examplefilenm):
    kwargs = torch_test_cases(examplefilenm)
    
    pretrained_model_filenm = get_pretrained_modelfilenm(examplefilenm)
    if not pretrained_model_filenm or not os.path.exists(pretrained_model_filenm):
        dgbml.doTrain(examplefilenm, **kwargs)
        pretrained_model_filenm = kwargs['outnm']

    kwargs['type'] = dgbml.TrainType.Transfer
    kwargs['modelin'] = pretrained_model_filenm
    kwargs['outnm'] = f'torch_test_{get_filenm_from_path(examplefilenm)}_transfer.h5'
    assert dgbml.doTrain(examplefilenm, **kwargs) == True

@pytest.fixture(scope="session", autouse=True)
def cleanup_after_tests(request):
    # This fixture will run once for the entire test session
    # Setup: Nothing to do here
    # Tear down: Delete the ML models or any other resources after all tests have run
    def finalizer():
        print("\nCleaning up ML models...")
        for filename in os.listdir():
            if any(filename.endswith(ext) for ext in ['.h5', '.onnx', '.pt', '.pth']):
                os.remove(filename)
    request.addfinalizer(finalizer)


