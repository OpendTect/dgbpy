import sys
sys.path.insert(0, '..')

import os
import fnmatch, pytest
import dgbpy.keystr as dbk
import dgbpy.mlapply as dgbml
import dgbpy.dgbkeras as dgbkeras
import dgbpy.dgbtorch as dgbtorch
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

@pytest.mark.skipif(not dgbkeras.hasKeras(), reason="Keras is not available")
@pytest.mark.parametrize('examplefilenm', examples)
def test_doTrain_keras_new_trainingtype(examplefilenm):
    kwargs = keras_test_cases()
    assert dgbml.doTrain(examplefilenm, **kwargs) == True

@pytest.mark.skipif(not dgbkeras.hasKeras(), reason="Keras is not available")
@pytest.mark.parametrize('examplefilenm', examples)
def test_doTrain_keras_resume_trainingtype(examplefilenm):
    kwargs = keras_test_cases(examplefilenm)
    if not os.path.exists(kwargs['outnm']):
        dgbml.doTrain(examplefilenm, **kwargs)
    kwargs['modelin'] = kwargs['outnm']
    kwargs['type'] = dgbml.TrainType.Resume
    assert dgbml.doTrain(examplefilenm, **kwargs) == True

@pytest.mark.skipif(not dgbkeras.hasKeras(), reason="Keras is not available")
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

    
@pytest.mark.skipif(not dgbtorch.hasTorch(), reason="Torch is not available")
@pytest.mark.parametrize('examplefilenm', examples)
def test_doTrain_torch_new_trainingtype(examplefilenm):
    kwargs = torch_test_cases(examplefilenm)
    assert dgbml.doTrain(examplefilenm, **kwargs) == True
    models.append(kwargs['outnm'])

@pytest.mark.skipif(not dgbtorch.hasTorch(), reason="Torch is not available")
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

@pytest.mark.skipif(not dgbtorch.hasTorch(), reason="Torch is not available")
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




# getClasses used to re-derive the class list from each group (well) in turn,
# so the model was built for the last group's classes only.
import numpy as np
import dgbpy.mlio as dgbmlio

def _litho_info( classes ):
    info = get_loglog_info()
    info[dbk.classdictstr] = True
    info[dbk.classesdictstr] = list(classes)
    return info

def test_getClasses_keeps_example_file_classlist():
    info = _litho_info( [1,2,3,4,5] )
    wella = np.array( [[1],[2],[3],[4],[5]], dtype=np.single )
    wellb = np.array( [[1],[2],[1],[2]], dtype=np.single )
    for welly in (wella, wellb):
        dgbmlio.getClasses( info, [welly] )
        dgbmlio.normalize_class_vector( welly, info[dbk.classesdictstr] )
        assert len(info[dbk.classesdictstr]) == 5, \
            'a single group must not narrow the example file class list'
    nroutputs = len( info[dbk.classesdictstr] )
    assert int(np.concatenate([wella,wellb]).max()) < nroutputs
    assert np.array_equal( wella.ravel(), [0,1,2,3,4] )
    assert np.array_equal( wellb.ravel(), [0,1,0,1] )

def test_getClasses_still_derives_when_none_given():
    info = _litho_info( [] )
    y = np.array( [[1],[3],[5]], dtype=np.single )
    dgbmlio.getClasses( info, [y] )
    assert list(info[dbk.classesdictstr]) == [1,3,5]

def test_checkClassVector_names_the_offending_value():
    dgbmlio.checkClassVector( np.array([0,1,2]), [1,2,3], 'y_train' )
    with pytest.raises( ValueError, match='outside the 2 classes' ):
        dgbmlio.checkClassVector( np.array([0,1,4]), [1,2], 'y_train' )

def test_checkFiniteData_accepts_clean_data():
    dgbmlio.checkFiniteData( np.array([1.0, -2.5, 0.0], dtype=np.single), 'x_train' )
    dgbmlio.checkFiniteData( np.array([1, 2, 3], dtype=np.int32), 'y_train' )

def test_checkFiniteData_rejects_opendtect_undef():
    arr = np.array( [1.0, 2.0, 1e30, 3.0], dtype=np.single )
    with pytest.raises( ValueError, match='OpendTect undefined' ):
        dgbmlio.checkFiniteData( arr, 'y_train' )

def test_checkFiniteData_rejects_nan_and_inf():
    for bad in (np.nan, np.inf, -np.inf):
        arr = np.array( [1.0, bad, 3.0], dtype=np.single )
        with pytest.raises( ValueError, match='NaN or infinite' ):
            dgbmlio.checkFiniteData( arr, 'x_train' )


# Clustering data has no target; getCubeLets returned an uninitialised
# placeholder for it, which checkFiniteData then rejected.
import h5py
import dgbpy.hdf5 as dgbhdf5

def _clustering_examplefile( tmp_path, nrpts=50, nrattribs=2, nrsamples=21 ):
    fnm = str( tmp_path / 'clustering.h5' )
    with h5py.File( fnm, 'w' ) as h5file:
        grp = h5file.create_group( 'Survey' ).create_group( 'Well' )
        grp.create_dataset( dbk.xdatadictstr,
                            data=np.random.rand(nrpts,nrattribs,nrsamples).astype(np.single) )
    info = get_default_info( nr_inattr=nrattribs )
    info[dbk.segmentdictstr] = True
    info[dbk.inpshapedictstr] = nrsamples
    info[dbk.filedictstr] = fnm
    return info, {'Well': list(range(nrpts))}

def test_getCubeLets_clustering_placeholder_target_is_defined(tmp_path):
    info, collection = _clustering_examplefile( tmp_path )
    ret = dgbhdf5.getCubeLets( info, collection, 'Survey' )
    assert ret[dbk.xtraindictstr].shape[0] == 50
    ytrain = ret[dbk.ytraindictstr]
    assert ytrain.shape[0] == 50
    assert np.all( np.isfinite(ytrain) ) and np.all( ytrain == 0 ), \
        'the unused target placeholder must not be uninitialised memory'

def test_getTrainingDataByInfo_skips_target_checks_for_clustering(tmp_path, monkeypatch):
    info, collection = _clustering_examplefile( tmp_path )
    dsets = dgbhdf5.getDatasets( info, {'Survey': collection}, validation=False )
    dsets[dbk.ytraindictstr][:] = dgbmlio.odundefval
    monkeypatch.setattr( dgbhdf5, 'getDatasets', lambda *a, **kw: dsets )
    ret = dgbmlio.getTrainingDataByInfo( info, {'Survey': collection} )
    assert ret[dbk.xtraindictstr].shape[0] == 50
    dsets[dbk.xtraindictstr][0] = np.nan
    with pytest.raises( ValueError, match='x_train' ):
        dgbmlio.getTrainingDataByInfo( info, {'Survey': collection} )
