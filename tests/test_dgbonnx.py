import sys
sys.path.insert(0, '..')

import os, pytest
import dgbpy.keystr as dbk
import dgbpy.dgbkeras as dgbkeras
import dgbpy.hdf5 as dgbhdf5
import dgbpy.dgbonnx as dgbonnx

from init_data import *
from models import make_onnx_model, get_model_filename

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


@pytest.mark.parametrize('data',
                         (get_2d_seismic_imgtoimg_data(nrclasses=5), get_3d_seismic_imgtoimg_data(nrclasses=5)),
                         ids=['2D_seismic_imgtoimg', '3D_seismic_imgto_img'])
def test_onnx_model_one_inattr_one_outattr(data):
    info = data[dbk.infodictstr]
    isclassification = info[dbk.classdictstr]
    inpshape = info[dbk.inpshapedictstr]
    outshape = info[dbk.outshapedictstr]
    if isinstance(inpshape, int): inpshape = [inpshape]
    if isinstance(outshape, int): outshape = [outshape]

    shapepar = inpshape[1:] if 1 in inpshape else inpshape
    make_onnx_model(shapepar, 1, 1, 'channels_first')
    modelfn = get_model_filename(shapepar)+'.onnx'

    model = dgbonnx.OnnxModel(modelfn)

    nroutputs = dgbhdf5.getNrOutputs(info)

    samples = data[dbk.xvaliddictstr]
    withpred = True
    withprobs = []
    withconfidence = False
    doprobabilities = len(withprobs) > 0

    dictinpshape = tuple( inpshape )
    dictoutshape = tuple( outshape )
    pred = dgbonnx.apply(model, info, samples, None, isclassification, withpred, withprobs, withconfidence, doprobabilities, dictinpshape, dictoutshape, nroutputs)
    prediction = pred[dbk.preddictstr]

    yvalid = data[dbk.yvaliddictstr]

    if getNrDims(info[dbk.inpshapedictstr]) < 3: 
        prediction = prediction[:, :, np.newaxis,...] # Because  apply will return (n,c,x,y) for 2D instead of (n,c,x,y,z)  

    assert prediction.shape == yvalid.shape, f"Prediction shape {prediction.shape} does not match yvalid shape {yvalid.shape}"

    os.remove(modelfn)




