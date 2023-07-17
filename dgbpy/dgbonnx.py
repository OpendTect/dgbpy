import numpy as np
import onnx
import onnxruntime as rt
import dgbpy.keystr as dgbkeys
import dgbpy.hdf5 as dgbhdf5
import odpy.hdf5 as odhdf5
import dgbpy.onnx_classes as oc


def get_model_shape( shape, nrattribs, attribfirst=True ):
    ret = ()
    if attribfirst:
        ret += (nrattribs,)
    if isinstance( shape, int ):
        ret += (shape,)
        if not attribfirst:
            ret += (nrattribs,)
        return ret
    else:
        for i in shape:
            if i > 1:
                ret += (i,)
    if attribfirst:
        if len(ret) == 1:
            ret += (1,)
    else:
        if len(ret) == 0:
            ret += (1,)
    if not attribfirst:
        ret += (nrattribs,)
    return ret

def get_output_shape( shape ):
    ret = ()
    if isinstance( shape, int ):
        ret += (shape,)
    else:
        for i in shape:
            if i > 1:
                ret += (i,)
    if len(ret) == 0:
        ret += (1,)
    return ret

def getModelDims( model_shape ):
    ret = model_shape[1:]
    if len(ret) == 1 and ret[0] == 1:
        return 0
    return len(ret)

def load( modelfnm ):
    model = None
    h5file = odhdf5.openFile( modelfnm, 'r' )
    modelgrp = h5file['model']
    savetype = odhdf5.getText(modelgrp, 'type')
    if savetype == dgbkeys.onnxplfnm:
        modfnm = odhdf5.getText( modelgrp, 'path' )
        modfnm = dgbhdf5.translateFnm( modfnm, modelfnm )
        model = OnnxModel(str(modfnm))
    h5file.close()
    return model

class OnnxModel():
    def __init__(self, filepath : str):
        self.name = filepath
        self.onnx_mdl = onnx.load(self.name)
        providers = [dgbkeys.onnxcudastr, dgbkeys.onnxcpustr]
        try:
            self.session = rt.InferenceSession(self.name, providers=providers)
        except RuntimeError:
            self.session = rt.InferenceSession(self.name, providers=[dgbkeys.onnxcpustr])

    def __call__(self, inputs):
        self.inputs = inputs
        ort_inputs = {self.session.get_inputs()[0].name: self.inputs}
        ort_outs = np.array(self.session.run(None, ort_inputs))[-1]
        return ort_outs

def apply( model, samples, scaler, isclassification, withpred, withprobs, withconfidence, doprobabilities, dictinpshape, dictoutshape, nroutputs):
    ret = {}
    res = None
    out_shape = get_output_shape( dictoutshape )
    img2img = len(out_shape) > 2

    predictions = []
    for input in samples:
        input = np.expand_dims(input, axis=0)
        pred = model(input)
        for _ in pred:
            predictions.append(_)

    predictions = np.array(predictions)
    if withpred:
        if isclassification:
            if not (doprobabilities or withconfidence):
                if predictions.shape[1] > 1:
                    res = np.argmax(predictions, axis=1)
                ret.update({dgbkeys.preddictstr: res})

        if not isinstance(res, np.ndarray):
            res = predictions
            ret.update({dgbkeys.preddictstr: res})

    if isclassification and (doprobabilities or withconfidence or withpred):
        if len(ret)<1:
            allprobs = (np.array(predictions)).transpose()
        else:
            allprobs = ret[dgbkeys.preddictstr]
        indices = None
        if withconfidence or not img2img or (img2img and nroutputs>2):
            N = 2
            if img2img:
                indices = np.argpartition(allprobs,-N,axis=1)[:,-N:]
            else:
                indices = np.argpartition(allprobs,-N,axis=0)[-N:]
        if withpred and isinstance( indices, np.ndarray ):
            if img2img:
                ret.update({dgbkeys.preddictstr: indices[:,-1:]})
            else:
                ret.update({dgbkeys.preddictstr: indices[-1:]})
        if doprobabilities and len(withprobs) > 0:
            res = np.copy(allprobs[withprobs])
            ret.update({dgbkeys.probadictstr: res})
        if withconfidence:
            N = 2
            predictions = np.array(predictions)
            x = predictions.shape[0]
            indices = np.argpartition(predictions.transpose(),-N,axis=0)[-N:].transpose()
            sortedprobs = predictions.transpose()[indices.ravel(),np.tile(np.arange(x),N)].reshape(N,x)
            res = np.diff(sortedprobs,axis=0)
            ret.update({dgbkeys.confdictstr: res})

    return ret


    

