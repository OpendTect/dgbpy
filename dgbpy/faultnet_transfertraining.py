#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
#

import json
import numpy as np
from os import path
from time import perf_counter

import torch

import dgbpy.keystr as dgbkeys
from dgbpy import hdf5 as dgbhdf5
from dgbpy import mlio as dgbmlio
from dgbpy.trainerbase import TrainerBase


class FaultNetTransferTrainer(TrainerBase):
    def __init__(self, modelfnm):
        super().__init__(modelfnm)
        self.gamma_ = 0.5
        self.tr_steps_ = 200
        self.learnrate_ = 1e-3
        self.crop_size_ = 256
        self.batch_size_ = 10
        self.microbatch_size_ = 1
        self.rng_ = np.random.default_rng()
        self.log_every_n_steps_ = 0
        self.metrics_ = None

        self.info_ = None
        self.device_ = torch.device("cpu")
        self.params_ = None
        self.model_ = None

        self.samples_ = []
        self.inference_input_ = None

    def setParameters(self, pars):
        if not isinstance(pars, dict):
            raise ValueError("Invalid train parameters: expected a JSON object.")

        trpars = pars.get("trainer_params")
        if isinstance(trpars, dict):
            usepars = dict(trpars)

        else:
            usepars = dict(pars)

        if "gamma" in usepars:
            self.gamma_ = float(usepars["gamma"])

        self.gamma_ = max(0.0, min(1.0, self.gamma_))

        if "tr_steps" in usepars:
            self.tr_steps_ = max(1, int(usepars["tr_steps"]))

        if "learnrate" in usepars:
            self.learnrate_ = float(usepars["learnrate"])

        if "crop_size" in usepars:
            self.crop_size_ = int(usepars["crop_size"])

        if "batch_size" in usepars:
            batch_size = usepars["batch_size"]
            if type(batch_size) is not int or batch_size < 1:
                raise ValueError("Invalid 'batch_size': expected a positive integer.")
            self.batch_size_ = batch_size

        if "microbatch_size" in usepars:
            microbatch_size = usepars["microbatch_size"]
            if type(microbatch_size) is not int or microbatch_size < 1:
                raise ValueError("Invalid 'microbatch_size': expected a positive integer.")
            self.microbatch_size_ = microbatch_size

        if self.crop_size_ < 16 or self.crop_size_ % 16 != 0:
            raise ValueError(
                "Invalid 'crop_size'. Fault-Net requires a positive cube dimension "
                "divisible by 16."
            )

        if "log_every_n_steps" in usepars:
            self.log_every_n_steps_ = max(0, int(usepars["log_every_n_steps"]))

        self.samples_ = []
        self.inference_input_ = None
        self.metrics_ = None
        self._prepare_model()

    def _prepare_model(self):
        _, info = dgbmlio.getModel(self.modelfnm_, fortrain=True, pars=None)
        platform = info.get(dgbkeys.plfdictstr, "")
        if platform == dgbkeys.onnxplfnm:
            platform = dgbkeys.torchplfnm

        if platform != dgbkeys.torchplfnm:
            raise ValueError(
                f"Unsupported training platform for faultnet_transfer trainer: {platform}. "
                "Only torch models are supported."
            )

        from dgbpy import dgbtorch
        trmodel = dgbtorch.load(self.modelfnm_, info)
        trmodel = dgbtorch.transfer(trmodel, info)

        self.info_ = info
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = trmodel.to(self.device_)
        self.params_ = dict(dgbtorch.torch_dict)
        self.params_["learnrate"] = self.learnrate_

    def nrArraysPerSample(self):
        return 2

    def addData(self, seismicarr, labelarr):
        seis = np.asarray(seismicarr, dtype=np.float32)
        labels = np.asarray(labelarr, dtype=np.float32)
        if seis.shape != labels.shape:
            raise ValueError(f"Shape mismatch between seismic {seis.shape} and labels {labels.shape}")

        seis = self._prepare_seismic(seis, self.crop_size_)
        labels = np.reshape(labels, seis.shape)
        if not np.all(np.isfinite(labels)):
            raise ValueError("Labels must contain only finite values.")
        
        fault_indices = np.flatnonzero(np.any(labels[0] > 0.0, axis=0))
        if fault_indices.size == 0:
            raise ValueError("Each training volume must contain at least one labelled fault.")

        self.samples_.append((seis, labels, fault_indices))
        self.inference_input_ = None
        
        self.metrics_ = None

    def InfereInputData(self, seismicarr):
        """Store normalized seismic-only input; spatial sizes must be multiples of 16."""
        if self.model_ is None:
            raise ValueError("No model available for inference. Initialize parameters first.")

        seis = self._prepare_seismic(seismicarr, 16)
        if any(dim % 16 != 0 for dim in seis.shape[2:]):
            raise ValueError(
                "Whole-volume Fault-Net inference requires each spatial dimension "
                f"to be divisible by 16. Got shape {seis.shape}."
            )

        self.inference_input_ = np.ascontiguousarray(seis)

    def _prepare_seismic(self, seismicarr, minsize):
        seis = self._normalize_input_shape(np.asarray(seismicarr, dtype=np.float32))
        if seis.ndim != 5 or any(dim < minsize for dim in seis.shape[2:]):
            raise ValueError(
                "Fault-Net requires (N,C,inline,crossline,z) data with each spatial "
                f"dimension >= {minsize}. Got shape {seis.shape}."
            )
        if seis.shape[0] != 1:
            raise ValueError("Each input must contain exactly one volume (N=1).")
        if seis.shape[1] == 0:
            raise ValueError("Seismic data must have a nonempty channel dimension.")
        if not np.all(np.isfinite(seis)):
            raise ValueError("Seismic must contain only finite values.")

        return self._normalize_seismic(seis)

    def _expected_input_rank(self):
        if not isinstance(self.info_, dict):
            return None

        inpshape = self.info_.get(dgbkeys.inpshapedictstr)
        if inpshape is None:
            return None

        if isinstance(inpshape, int):
            spatialdims = 3

        else:
            spatialdims = len(inpshape)

        return 2 + spatialdims

    def _normalize_sample_shape(self, seis, labels):
        seis = self._normalize_input_shape(seis)
        return seis, np.reshape(labels, seis.shape)

    def _normalize_input_shape(self, seis):
        expectedrank = self._expected_input_rank()
        if expectedrank is None:
            return seis

        if seis.ndim > expectedrank:
            raise ValueError(
                f"Seismic rank {seis.ndim} exceeds expected model input rank {expectedrank}."
            )

        if seis.ndim < expectedrank:
            addaxes = expectedrank - seis.ndim
            newshape = (1,) * addaxes + tuple(seis.shape)
            seis = np.reshape(seis, newshape)

        nrattribs = dgbhdf5.getNrAttribs(self.info_) if isinstance(self.info_, dict) else 1
        if seis.shape[1] not in (1, nrattribs):
            raise ValueError(
                f"Unexpected channel dimension {seis.shape[1]} for model expecting {nrattribs} attribute(s). "
                f"Prepared seismic shape: {seis.shape}."
            )

        return seis

    @staticmethod
    def _normalize_seismic(seis):
        # Min-max scaling to [0,1], as done by the original Fault-Net inference code
        smin = float(np.min(seis))
        srange = float(np.max(seis)) - smin
        if srange <= 0.0:
            return np.full_like(seis, 0.5)

        return ((seis - smin) / srange).astype(np.float32, copy=False)

    def _crop_and_augment(self, sample):
        seis, labels, fault_indices = sample
        if (True):
            shape = seis.shape[2:]
            anchor = np.unravel_index(self.rng_.choice(fault_indices), shape)
            slices = [slice(None), slice(None)]
            for pos, dim in zip(anchor, shape):
                first = max(0, pos - self.crop_size_ + 1)
                last = min(pos, dim - self.crop_size_)
                start = int(self.rng_.integers(first, last + 1))
                slices.append(slice(start, start + self.crop_size_))
            
            seis, labels = seis[tuple(slices)], labels[tuple(slices)]
            # Preserve the vertical axis; transform seismic and supervision together.
            turns = int(self.rng_.integers(4))
            seis = np.rot90(seis, turns, axes=(2, 3))
            labels = np.rot90(labels, turns, axes=(2, 3))
            if (False):
                for axis in (2, 3):
                    if self.rng_.integers(2):
                        seis = np.flip(seis, axis=axis)
                        labels = np.flip(labels, axis=axis)
        
        return (
            torch.from_numpy(np.ascontiguousarray(seis)).to(self.device_),
            torch.from_numpy(np.ascontiguousarray(labels)).to(self.device_),
        )

    def _prepare_batch(self, first, count):
        seis, labels = self._crop_and_augment(self.samples_[first % len(self.samples_)])
        if count == 1:
            return seis, labels
        inputs = seis.new_empty((count, *seis.shape[1:]))
        targets = labels.new_empty((count, *labels.shape[1:]))
        inputs[0] = seis[0]
        targets[0] = labels[0]
        for idx in range(1, count):
            seis, labels = self._crop_and_augment(
                self.samples_[(first + idx) % len(self.samples_)]
            )
            inputs[idx] = seis[0]
            targets[idx] = labels[0]
        return inputs, targets

    def hasSamples(self):
        return len(self.samples_) > 0

    def _prediction_loss(self, seis, lab):
        pred = self.model_(seis)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]

        if pred.dim() == lab.dim() + 1:
            chidx = 1 if pred.size(1) > 1 else 0
            pred = pred[:, chidx, ...]
            lab = lab[:, 0, ...] if lab.dim() == pred.dim() + 1 else lab

        elif lab.dim() == pred.dim() + 1:
            lab = lab[:, 0, ...]

        y = (lab > 0.0).float()
        mask = (lab >= 0.0).float()
        p = torch.clamp(pred, 0.0, 1.0)
        dims = tuple(range(1, p.ndim))
        numer = torch.sum(p * y * mask, dim=dims)
        den = (self.gamma_ * torch.sum(mask * y, dim=dims)
               + (1.0 - self.gamma_) * torch.sum(mask * p, dim=dims))
        return (1.0 - numer / (den + 1e-8)).mean()

    def _torch_step(self, step, optimizer):
        self.model_.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), dtype=torch.float32, device=self.device_)
        first = step * self.batch_size_
        for offset in range(0, self.batch_size_, self.microbatch_size_):
            count = min(self.microbatch_size_, self.batch_size_ - offset)
            seis, labels = self._prepare_batch(first + offset, count)
            loss = self._prediction_loss(seis, labels)
            weight = count / self.batch_size_
            (loss * weight).backward()
            total_loss += loss.detach() * weight
            del loss, seis, labels
        optimizer.step()

        return total_loss

    def _fixed_eval_sample(self):
        seis, labels, fault_indices = self.samples_[0]
        shape = seis.shape[2:]
        anchor = np.unravel_index(fault_indices[0], shape)
        slices = [slice(None), slice(None)]
        for pos, dim in zip(anchor, shape):
            start = min(max(0, pos - self.crop_size_ // 2), dim - self.crop_size_)
            slices.append(slice(start, start + self.crop_size_))
        return seis[tuple(slices)], labels[tuple(slices)]

    def evaluate(self, sample=None):
        """Return four rotation losses on a fixed training crop, not held-out validation."""
        if not self.samples_:
            raise ValueError("No training samples available for evaluation.")
        if sample is None:
            sample = self._fixed_eval_sample()
        seis, labels = sample
        modes = [(module, module.training) for module in self.model_.modules()]
        losses = {}
        self.model_.eval()
        try:
            with torch.no_grad():
                for turns in range(4):
                    inp = torch.from_numpy(np.ascontiguousarray(
                        np.rot90(seis, turns, axes=(2, 3))
                    )).to(self.device_)
                    lab = torch.from_numpy(np.ascontiguousarray(
                        np.rot90(labels, turns, axes=(2, 3))
                    )).to(self.device_)
                    losses[f"rot{turns * 90}"] = float(
                        self._prediction_loss(inp, lab).cpu().item()
                    )
        finally:
            for module, training in modes:
                module.training = training
        losses["mean"] = sum(losses.values()) / 4
        return losses

    def _log_evaluation(self, step, sample):
        losses = self.evaluate(sample)
        print(
            f"[faultnet_eval] step={step} "
            + " ".join(f"{key}={value:.6f}" for key, value in losses.items()),
            flush=True,
        )
        return {"step": step, **losses}

    def _train(self):
        if not self.samples_:
            raise ValueError("No training samples were provided before finalize.")

        trpars = [par for par in self.model_.parameters() if par.requires_grad]
        if not trpars:
            raise ValueError("No trainable parameters available in transfer model.")

        started = perf_counter()
        optimizer = torch.optim.Adam(trpars, lr=self.learnrate_)
        loss_sum = torch.zeros((), dtype=torch.float32, device=self.device_)
        eval_sample = self._fixed_eval_sample()
        self.eval_history_ = [self._log_evaluation(0, eval_sample)]
        for istep in range(self.tr_steps_):
            losst = self._torch_step(istep, optimizer)
            loss_sum = loss_sum + losst
            if self.log_every_n_steps_ > 0 and ((istep + 1) % self.log_every_n_steps_ == 0):
                print(
                    f"[faultnet_transfer] step={istep + 1}/{self.tr_steps_} "
                    f"loss={float(losst.cpu().item()):.6f} "
                    f"elapsed={perf_counter() - started:.1f}s",
                    flush=True,
                )
            if (istep + 1) % 200 == 0 or istep + 1 == self.tr_steps_:
                self.eval_history_.append(self._log_evaluation(istep + 1, eval_sample))

        return float((loss_sum / self.tr_steps_).cpu().item())

    def finalize(self):
        self.metrics_ = None
        try:
            avgloss = self._train()
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                "CUDA out of memory during Fault-Net training "
                f"(crop_size={self.crop_size_}, batch_size={self.batch_size_}, "
                f"microbatch_size={self.microbatch_size_}). "
                "Reduce microbatch_size or crop_size before restarting training."
            ) from exc
        self.metrics_ = {
            "trainer_id": "faultnet_transfer",
            "samples": len(self.samples_),
            "tr_steps": self.tr_steps_,
            "avg_masked_dice_loss": avgloss,
            "gamma": self.gamma_,
            "crop_size": self.crop_size_,
            "batch_size": self.batch_size_,
            "microbatch_size": self.microbatch_size_,
            "platform": dgbkeys.torchplfnm,
            "training_crop_evaluation": self.eval_history_,
        }

        return self.metrics_

    def infere(self):
        """Infer explicit seismic-only input, or the stored volume after training."""
        if self.model_ is None:
            raise ValueError("No model available for inference. Initialize parameters first.")

        seis = self.inference_input_
        if seis is None:
            if self.metrics_ is None:
                raise ValueError(
                    "No inference input available. Call InfereInputData or finalize first."
                )
            if len(self.samples_) != 1:
                raise ValueError("Whole-volume inference requires exactly one stored input volume.")
            seis = self.samples_[0][0]
        modes = [(module, module.training) for module in self.model_.modules()]
        self.model_.eval()
        try:
            with torch.no_grad():
                inp = torch.from_numpy(np.ascontiguousarray(seis)).to(self.device_)
                pred = self.model_(inp)
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]
        
                if not isinstance(pred, torch.Tensor):
                    raise TypeError("Fault-Net inference must return a tensor.")
        
                if tuple(pred.shape) == (1, 1, *seis.shape[2:]):
                    pred = pred[0, 0]
        
                elif tuple(pred.shape) == (1, *seis.shape[2:]):
                    pred = pred[0]
        
                else:
                    raise ValueError(
                        f"Unexpected Fault-Net inference shape {tuple(pred.shape)}; "
                        f"expected (1,1,{','.join(map(str, seis.shape[2:]))})."
                    )
        
                if not torch.isfinite(pred).all().item():
                    raise ValueError("Fault-Net inference produced nonfinite predictions.")
        
                finalinference = torch.clamp(pred, 0.0, 1.0).to( device="cpu", dtype=torch.float32 ).numpy()
                return np.ascontiguousarray(finalinference)
        
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                "CUDA out of memory during whole-volume Fault-Net inference "
                f"for input shape {seis.shape}. No tiling or fallback is enabled.") from exc
        
        finally:
            for module, training in modes:
                module.training = training

    def save(self, targetdir, outputname):
        if self.metrics_ is None:
            raise ValueError("No trained model available to save. Call finalize first.")

        import shutil
        import h5py
        import onnx
        import odpy.hdf5 as odhdf5
        from onnx import numpy_helper
        from onnx2torch.onnx_graph import OnnxGraph

        outfnm = path.abspath(path.join(targetdir, outputname))
        if not outfnm.endswith(".h5"):
            outfnm += ".h5"
        outonnx = path.splitext(outfnm)[0] + ".onnx"

        with h5py.File(self.modelfnm_, "r") as baseh5:
            modelgrp = baseh5["model"]
            if odhdf5.getText(modelgrp, "type") != "onnx":
                raise ValueError(
                    "Fault-Net transfer save requires an ONNX base model."
                )
            baseonnx = dgbhdf5.translateFnm(
                odhdf5.getText(modelgrp, "path"), self.modelfnm_
            )

        onnxmodel = onnx.load(str(baseonnx))
        graph = OnnxGraph(onnxmodel.graph)
        initializers = {tensor.name: tensor for tensor in onnxmodel.graph.initializer}
        parameters = dict(self.model_.named_parameters())
        mappedparameters = set()
        replacements = {}

        for modulename, node in graph.nodes.items():
            if node.operation_type != "Conv":
                continue

            module = self.model_.get_submodule(modulename)
            if not isinstance(module, torch.nn.Conv3d):
                raise TypeError(
                    f"Expected Conv3d module for ONNX node {modulename}; "
                    f"got {type(module).__name__}."
                )

            for parametername, inputindex in (("weight", 1), ("bias", 2)):
                parameter = getattr(module, parametername)
                if parameter is None:
                    continue

                fullname = f"{modulename}.{parametername}"
                if fullname not in parameters:
                    raise ValueError(f"Unmapped model parameter: {fullname}.")
                if inputindex >= len(node.input_values):
                    raise ValueError(f"Missing ONNX input for {fullname}.")

                initializername = node.input_values[inputindex]
                if initializername not in initializers:
                    raise ValueError(
                        f"Missing ONNX initializer {initializername} for {fullname}."
                    )

                original = numpy_helper.to_array(initializers[initializername])
                updated = parameter.detach().cpu().numpy()
                if updated.shape != original.shape:
                    raise ValueError(
                        f"Shape mismatch for {fullname}: "
                        f"{updated.shape} versus {original.shape}."
                    )
                if updated.dtype != original.dtype:
                    raise ValueError(
                        f"Datatype mismatch for {fullname}: "
                        f"{updated.dtype} versus {original.dtype}."
                    )
                if not np.all(np.isfinite(updated)):
                    raise ValueError(f"Nonfinite values in {fullname}.")

                previous = replacements.get(initializername)
                if previous is not None and not np.array_equal(previous, updated):
                    raise ValueError(
                        f"Conflicting updates for shared initializer {initializername}."
                    )

                replacements[initializername] = updated.copy()
                mappedparameters.add(fullname)

        unmapped = sorted(set(parameters) - mappedparameters)
        if unmapped:
            raise ValueError(
                "Fault-Net save cannot map these parameters: " + ", ".join(unmapped)
            )

        for initializername, updated in replacements.items():
            initializers[initializername].CopyFrom(
                numpy_helper.from_array(updated, name=initializername)
            )

        onnx.checker.check_model(onnxmodel)
        onnx.save(onnxmodel, outonnx)

        modes = [(module, module.training) for module in self.model_.modules()]
        try:
            shutil.copy2(self.modelfnm_, outfnm)
            with h5py.File(outfnm, "r+") as outputh5:
                outputh5["model"].attrs["path"] = np.bytes_(outonnx)
                outputh5.attrs["training_config"] = np.bytes_(
                    json.dumps(self.params_)
                )
        finally:
            self.model_.to(self.device_)
            for module, training in modes:
                module.training = training

        #_________________---------------SAVE METRICS FOR EVALUATION GRAPHICS (LOSS PLOT)
        # metricsfnm = path.join(targetdir, f"{outputname}_transfertraining_metrics.json")
        # with open(metricsfnm, "w", encoding="utf-8") as fobj:
        #     json.dump(self.metrics_, fobj, indent=2)
        #_________________---------------SAVE METRICS FOR EVALUATION GRAPHICS (LOSS PLOT)
        
        return outfnm
