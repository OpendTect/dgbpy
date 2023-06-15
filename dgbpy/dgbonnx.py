import numpy as np
import onnxruntime as rt
import dgbpy.keystr as dgbkeys
import dgbpy.hdf5 as dgbhdf5
import odpy.hdf5 as odhdf5


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

def apply( model, info, samples, scaler, isclassification, withpred, withprobs, withconfidence, doprobabilities ):
    ret = {}
    res = None
    attribs = dgbhdf5.getNrAttribs(info)
    model_shape = get_model_shape(info[dgbkeys.inpshapedictstr], attribs, True)
    ndims = getModelDims(model_shape)

    try:
        img2img = info[dgbkeys.seisimgtoimgtypestr]
        img2img = True
    except KeyError:
        img2img = False

    if isclassification:
        nroutputs = len(info[dgbkeys.classesdictstr])
    else:
        nroutputs = dgbhdf5.getNrOutputs(info)

    predictions = []
    predictions_prob = []
    for input in samples:
        pred = model(input)
        pred_prob = pred.copy()
        if isclassification:
            pred = np.argmax(pred, axis=1)
        for _ in pred:
            predictions.append(_)
        for _prob in pred_prob:
            predictions_prob.append(_prob)

    if withpred:
        if isclassification:
            if not (doprobabilities or withconfidence):
                try:
                    res = np.array(predictions_prob)
                    res = np.transpose(np.array(predictions))
                except AttributeError:
                    pass

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
            predictions_prob = np.array(predictions_prob)
            x = predictions_prob.shape[0]
            indices = np.argpartition(predictions_prob.transpose(),-N,axis=0)[-N:].transpose()
            sortedprobs = predictions_prob.transpose()[indices.ravel(),np.tile(np.arange(x),N)].reshape(N,x)
            res = np.diff(sortedprobs,axis=0)
            ret.update({dgbkeys.confdictstr: res})

    if withpred:
        res = np.transpose(np.array(predictions))
        ret.update({dgbkeys.preddictstr: res})

    if doprobabilities:
        res = np.transpose( np.array(predictions_prob) )
        ret.update({dgbkeys.probadictstr: res})
    if info[dgbkeys.learntypedictstr] == dgbkeys.seisimgtoimgtypestr:
        if ndims==3:
            if isclassification:
                ret[dgbkeys.preddictstr] = ret[dgbkeys.preddictstr].transpose(3, 2, 1, 0)
            else:
                ret[dgbkeys.preddictstr] = ret[dgbkeys.preddictstr].transpose(4, 3, 2, 1, 0)
        elif ndims==2:
            if isclassification:
                ret[dgbkeys.preddictstr] = ret[dgbkeys.preddictstr].transpose(2, 1, 0)
            else:
                ret[dgbkeys.preddictstr] = ret[dgbkeys.preddictstr].transpose(3, 2, 1, 0)
    return ret


    

