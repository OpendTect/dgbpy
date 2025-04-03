#__________________________________________________________________________
#
# Copyright:	(C) 1995-2022 dGB Beheer B.V.
# License:	Apache Version 2
#
#________________________________________________________________________
#
from abc import ABC, abstractmethod
import json
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
import sys
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum
import dgbpy.keystr as dgbkeys
import dgbpy.hdf5 as dgbhdf5
import odpy.hdf5 as odhdf5


class PlatformType(StrEnum):
    Keras =     dgbkeys.kerasplfnm
    Sklearn =   dgbkeys.scikitplfnm
    PyTorch =   dgbkeys.torchplfnm
    Onnx =      dgbkeys.onnxplfnm

class LearnType(StrEnum):
    SeisClass =     dgbkeys.seisclasstypestr
    SeisLogPred =   dgbkeys.seisproptypestr
    LogLogPred =    dgbkeys.loglogtypestr
    LogCluster =    dgbkeys.logclustertypestr
    SeisImg2Img =   dgbkeys.seisimgtoimgtypestr

class PredType(StrEnum):
    Continuous =        dgbkeys.continuousvalstr
    Classification =    dgbkeys.classdatavalstr
    Segments =          dgbkeys.segmenttypestr
    Unknown =           'Undetermined'

class ScalingType(StrEnum):
    GlobalStd =         dgbkeys.globalstdtypestr
    LocalStd =          dgbkeys.localstdtypestr
    LocalNormalized =   dgbkeys.normalizetypestr    # [0, 1]
    LocalMinMax =       dgbkeys.minmaxtypestr       # [0, 255]
    RangeStd =          dgbkeys.rangestdtypestr     #[-1, 1]

@dataclass
class ZipModelInfo():
    """Summary information for an external model."""
    pred_type: PredType     = PredType.Continuous
    platform: PlatformType  = PlatformType.PyTorch
    learn_type: LearnType   = LearnType.SeisImg2Img
    scale_type: ScalingType = ScalingType.GlobalStd
    scale_output: bool      = False
    input_names: [str]      = field(default_factory=list)
    output_names: [str]     = field(default_factory=list)
    input_shape: [int]      = field(default_factory=list) # use 0 to indicate dynamic axis size
    output_shape: [int]     = field(default_factory=list) # use 0 to indicate dynamic axis size
    data_format: str        = field(default='channels_first')
    model_version: str      = field(default='unknown')
    model_name: str         = field(default='unknown')
    platform_version: str   = field(default='unknown')

    def info(self) -> dict:
        return json.dumps({
                    'platform': self.platform.__str__(),
                    'pred_type': self.pred_type.__str__(),
                    'learn_type': self.learn_type.__str__(),
                    'scale_type': self.scale_type.__str__(),
                    'scale_output': self.scale_output,
                    'num_inputs': len(self.input_names),
                    'num_outputs': len(self.output_names),
                    'input_names': self.input_names,
                    'output_names': self.output_names,
                    'input_shape': [shp if shp else 0 for shp in self.input_shape] ,
                    'output_shape': [shp if shp else 0 for shp in self.output_shape],
                    'data_format': self.data_format,
                    'model_version': self.model_version,
                    'model_name': self.model_name,
                    'platform_version': self.platform_version
        })

class ZipPredictModel(ABC):
    """Abstract base class for ZipModel format machine learning models for prediction only

    This module defines the interface for ZipModel packaging of machine learning models
    for import into OpendTect for prediction only

    """
    modelinfo = None

    @abstractmethod
    def __init__(self, params: dict = {}) -> None:
        """Create a ZipModel instance by adding any deserialization and initialization code for the model."""
        raise NotImplementedError()  # pragma: no cover

    def __repr__(self) -> str:
        """Return a string representing the model object."""
        return self.__class__.__name__

    @abstractmethod
    def predict(self, model_input: npt.ArrayLike, params: dict = {}) -> np.ndarray:
        """Prediction method

        Parameters
        ----------
        model_input A ZipModel compatible input for the model to evaluate.
        params      Additional parameters to pass to the model for inference.

        Returns
        -------
        Model predictions

        """
        raise NotImplementedError()  # pragma: no cover

    @classmethod
    def info(clss) -> str:
        """Get information about the ZipModel

        Returns
        -------
        JSON formatted str

        """
        return clss.modelinfo.info() if clss.modelinfo else {}

class ZipTrainableModel(ZipPredictModel):
    """Abstract base class for ZipModel format machine learning models that can be used for prediction and training

    This module defines the interface for ZipModel packaging of machine learning models
    for import into OpendTect for both training and prediction

    """
    @abstractmethod
    def __init__(self, params: dict = {}) -> None:
        """Create a ZipModel instance by adding any deserialization and initialization code for the model."""
        raise NotImplementedError()  # pragma: no cover

    def saveas(self, fname: str) -> None:
        """Save the model artifacts

        Parameters
        ----------
        fname: str       the filename to save to
        """
        pass

    @abstractmethod
    def train_one_epoch(self, ):
        """Abstract method defining what the model does during a single training step

        Parameters
        ----------

        Returns
        -------

        """
        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def _dosave(self, fname:str) -> None:
        """

        """
        raise NotImplementedError()  # pragma: no cover

def load(modelfnm: str):
    model = None
    if dgbhdf5.isZipModelFile(modelfnm):
        model = load_modelimpl(modelfnm)
        return model
    try:
        h5file = odhdf5.openFile( modelfnm, 'r' )
        modelgrp = h5file['model']
        savetype = odhdf5.getText(modelgrp, 'type')
        if savetype == dgbkeys.zipmodelstr:
            modfnm = odhdf5.getText( modelgrp, 'path' )
            modfnm = dgbhdf5.translateFnm( modfnm, modelfnm )
            model = load_modelimpl(modfnm)
        else:
            raise ValueError(f"File {modelfnm} is not a valid ZipModel file")
    except Exception as e:
        raise e
    finally:
        h5file.close()
    return model

def load_modelimpl (modelfnm: str):
    model = None
    sys.path.insert(0, str(modelfnm))
    from zipmodel import ZipModel
    model = ZipModel()
    return model

def apply( model, infos, samples, scaler, isclassification, withpred, withprobs, withconfidence, doprobabilities, dictinpshape, dictoutshape, nroutputs ):
    ret = {}
    res = None
    img2img = dgbhdf5.isImg2Img(infos)
    nroutputs = dgbhdf5.getNrOutputs(infos)

    predictions = []
    predictions = model.predict(samples)
    if withpred:
        if isclassification:
            if not (doprobabilities or withconfidence):
                if nroutputs > 2:
                    predictions = np.argmax(predictions, axis=1)
                if nroutputs == 2:
                    predictions = predictions[:, -1]

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

