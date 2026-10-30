#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
#
# Interface that every trainer used by the deep learning train server must implement
#

from abc import ABC, abstractmethod
import numpy as np


class TrainerBase(ABC):
    def __init__(self, modelfnm):
        self.modelfnm_ = modelfnm

    @abstractmethod
    def setParameters(self, pars) -> None:
        """Reads the parameters of the training request (JSON dict) and prepares the model"""

    @abstractmethod
    def nrArraysPerSample(self) -> int:
        """Returns the number of arrays that make up one training sample"""

    @abstractmethod
    def addData(self, *arrays: np.ndarray) -> None:
        """Adds one training sample, made of nrArraysPerSample() arrays
           in the order in which they were sent by the client"""

    @abstractmethod
    def hasSamples(self) -> bool:
        """Returns True if at least one (input,label) pair was added"""

    @abstractmethod
    def finalize(self) -> dict:
        """Runs the training and returns a dict with the training metrics"""

    @abstractmethod
    def save(self, targetdir: str, outputname: str) -> str:
        """Saves the trained model as 'outputname' in 'targetdir'.
           Returns the full path of the saved model file"""
