#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : M. Fazelzadeh
# DATE     : Oct 2026
#
# Deep learning train server
#
#

import io
import json
import numpy as np
import os
import psutil
import selectors
import struct
import sys
import traceback as tb

from dgbpy.trainerbase import TrainerBase

# Trainer factories for different types of deep learning models
# Trainers must be classes inheriting from TrainerBase
from dgbpy import faultnet_transfertraining as fntrainer

class ParentExitCommand(Exception):
    pass
class ExitCommand(Exception):
    pass

# Add new trainer factories here. 
# the key should match the "trainer_id" used in the JSON file
# created by DeepLearning::TrainAccess::StartData
_trainers_factory = {
    'faultnet_transfer': fntrainer.FaultNetTransferTrainer,
}

def create_trainer(trainerid, modelfnm):
    trainer = _trainers_factory.get(trainerid)
    if trainer is None:
        known = ', '.join(sorted(_trainers_factory.keys()))
        raise ValueError(f'Unsupported trainer_id: {trainerid}. Known trainers: {known}')
    
    if not (isinstance(trainer, type) and issubclass(trainer, TrainerBase)):
        raise TypeError(f'Trainer "{trainerid}" must be a class inheriting from TrainerBase')
    
    return trainer(modelfnm)

class ModelTrainer:
    def __init__(self, modelfnm):
        self.modelfnm_ = modelfnm
        self.trainerid_ = None
        self.trainer_ = None
        self.pending_arrays_ = []

    def setParameters(self, pars):
        if not isinstance(pars, dict):
            raise ValueError("Invalid train parameters: expected a JSON object.")

        self.trainerid_ = pars.get("trainer_id")
        if not self.trainerid_:
            raise ValueError("Missing required train parameter: 'trainer_id'.")

        self.trainer_ = create_trainer(self.trainerid_, self.modelfnm_)
        self.pending_arrays_ = []
        self.trainer_.setParameters(pars)

    def addData(self, arrays):
        if self.trainer_ is None:
            raise ValueError("Training parameters were not initialized.")
        
        if not arrays:
            raise ValueError("No arrays received.")

        # A sample can arrive in one request or spread over several ones
        nrarrays = self.trainer_.nrArraysPerSample()
        if type(nrarrays) is not int or nrarrays <= 0:
            raise ValueError(f"Invalid return type or value for nrArraysPerSample()")
        
        for arr in arrays:
            self.pending_arrays_.append(arr)
            if len(self.pending_arrays_) == nrarrays:
                sample = self.pending_arrays_
                self.pending_arrays_ = []
                self.trainer_.addData(*sample)

    def finalize(self):
        if self.trainer_ is None:
            raise ValueError("Training parameters were not initialized.")
        
        if self.pending_arrays_:
            raise ValueError(f"Training data is incomplete: received "
                             f"{len(self.pending_arrays_)} of "
                             f"{self.trainer_.nrArraysPerSample()} arrays of the last sample.")

        has_samples = self.trainer_.hasSamples()
        if type(has_samples) is not bool:
            raise ValueError(f"Invalid return type from hasSamples()")

        if not has_samples:
            raise ValueError("No training samples were provided before finalize. "
                             "Please send at least one complete sample first.")
        
        return self.trainer_.finalize()

    def save(self, targetdir, outputname):
        if self.trainer_ is None:
            raise ValueError("Training parameters were not initialized.")
        
        if not isinstance(targetdir, str) or not targetdir or not os.path.isdir(targetdir):
            raise ValueError(f"Output directory '{targetdir}' does not exist or is not a directory.")
        if not isinstance(outputname, str) or not outputname.strip():
            raise ValueError("Missing or invalid 'output_model_name': expected a nonempty string.")

        saved_path = self.trainer_.save(os.path.abspath(targetdir), outputname)
        if not saved_path or not os.path.isfile(saved_path):
            raise ValueError(f"Failed to save the model to '{saved_path}'.")

        return saved_path


class Message:
    def __init__(self, selector, sock, addr, trainer):
        self._selector = selector
        self._sock = sock
        self._addr = addr
        self._recv_buffer = b""
        self._send_buffer = b""
        self._payload_len = None
        self._reqid = None
        self._subid = None
        self._jsonheader_len = None
        self._jsonheader = None
        self._request = None
        self._response_created = False
        self._serverexception = None
        self._killreq = False
        self._trainer = trainer

    def _set_selector_events_mask(self, mode):
        if mode == "r":
            events = selectors.EVENT_READ

        elif mode == "w":
            events = selectors.EVENT_WRITE

        elif mode == "rw":
            events = selectors.EVENT_READ | selectors.EVENT_WRITE

        else:
            raise ValueError(f"Invalid events mask mode {repr(mode)}.")
        
        self._selector.modify(self._sock, events, data=self)

    def _read(self):
        try:
            data = self._sock.recv(16777216)

        except BlockingIOError:
            return True
        
        if not data:
            self.close()
            return False
        
        self._recv_buffer += data
        return True

    def _write(self):
        if self._send_buffer:
            try:
                sent = self._sock.send(self._send_buffer)

            except BlockingIOError:
                return
            
            self._send_buffer = self._send_buffer[sent:]
            if sent and not self._send_buffer:
                self.close()

    def _json_encode(self, obj, encoding):
        json_hdr = json.dumps(obj, ensure_ascii=False).encode(encoding)
        return struct.pack("=i", len(json_hdr)) + json_hdr

    def _json_decode(self, json_bytes, encoding):
        hdrlen = 4
        json_hdr = struct.unpack("=i", json_bytes[:hdrlen])[0]
        tiow = io.TextIOWrapper(io.BytesIO(json_bytes[hdrlen:hdrlen + json_hdr]),
                                encoding=encoding, newline="")
        obj = json.load(tiow)
        tiow.close()
        return (json_hdr, obj, json_bytes[hdrlen + json_hdr:])

    def _array_decode(self, arrptr, shapes, dtypes):
        offset = 0
        arrs = []
        for shape, dtype in zip(shapes, dtypes):
            nrsamples = np.prod(shape, dtype=np.int64)
            arr = np.frombuffer(arrptr, dtype=dtype, count=nrsamples, offset=offset)
            arr = arr.reshape(shape)
            offset += arr.nbytes
            arrs.append(arr)

        return {"action": "data", "data": arrs}

    def _create_message(self, *, content_bytes, content_type, content_encoding, arrsize):
        jsonheader = {
            "byteorder": sys.byteorder,
            "content-type": content_type,
            "content-encoding": content_encoding,
            "content-length": len(content_bytes),
        }
        
        if arrsize is not None:
            jsonheader.update({"array-shape": arrsize})

        jsonheader_bytes = self._json_encode(jsonheader, "utf-8")
        payload = jsonheader_bytes + content_bytes
        od_hdr = struct.pack("=i", len(payload)) + struct.pack("=i", self._reqid) \
                 + struct.pack("=h", self._subid)
        
        return od_hdr + payload

    def _make_exception_report(self, msg, exc):
        _, _, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        stackstr = "".join(tb.extract_tb(exc_tb, limit=10).format())
        return f"{msg}:\n{repr(exc)} on line {exc_tb.tb_lineno} of script {fname}\n{stackstr}"

    def _create_response_json_content(self):
        action = self._request.get("action")
        content = {"result": None}
        if action == "status":
            content["result"] = "Server online"
            content["pid"] = psutil.Process().pid

        elif action == "kill":
            content["result"] = "Kill request received"
            self._killreq = True

        elif action == "parameters":
            try:
                self._trainer.setParameters(self._request.get("value"))

            except Exception as exc:
                self._serverexception = self._make_exception_report("Start error exception", exc)
                content["result"] = self._serverexception

            else:
                content["result"] = "Train parameters received"

        elif action == "finalize":
            try:
                metrics = self._trainer.finalize()
                content["result"] = "Training finalized"
                content["metrics"] = metrics

            except Exception as exc:
                self._serverexception = self._make_exception_report("Finalize error exception", exc)
                content["result"] = self._serverexception

        elif action == "save":
            try:
                val = self._request.get("value") or {}
                if not isinstance(val, dict):
                    raise ValueError("Invalid save request: expected a JSON object in 'value'.")
                outnm = val.get("output_model_name")
                targetdir = val.get("target_dir")
                outfnm = self._trainer.save(targetdir, outnm)
                content["result"] = "Training model saved"
                content["output_file"] = outfnm

            except Exception as exc:
                self._serverexception = self._make_exception_report("Save error exception", exc)
                content["result"] = self._serverexception

        else:
            content["result"] = f'Error: invalid action "{action}".'

        return {
            "content_bytes": self._json_encode(content, "utf-8"),
            "content_type": "text/json",
            "content_encoding": "utf-8",
            "arrsize": None,
        }

    def _create_response_array_content(self):
        content = {"result": None}
        try:
            self._trainer.addData(self._request.get("data"))
            content["result"] = "Train data received"

        except Exception as exc:
            self._serverexception = self._make_exception_report("Train data exception", exc)
            content["result"] = self._serverexception

        return {
            "content_bytes": self._json_encode(content, "utf-8"),
            "content_type": "text/json",
            "content_encoding": "utf-8",
            "arrsize": None,
        }

    def process_events(self, mask):
        if mask & selectors.EVENT_READ:
            self.read()

        if mask & selectors.EVENT_WRITE:
            self.write()

    def read(self):
        if not self._read():
            return
        
        if self._payload_len is None:
            self.process_odheader()

        if self._payload_len is not None and self._jsonheader is None:
            self.process_jsonheader()
        
        if self._jsonheader and self._request is None:
            self.process_request()

    def write(self):
        if self._request is not None and not self._response_created:
            self.create_response()

        self._write()

    def close(self):
        try:
            self._selector.unregister(self._sock)

        except Exception:
            pass

        try:
            self._sock.close()

        except Exception:
            pass

        self._sock = None

    def process_odheader(self):
        hdrlen = 10
        if len(self._recv_buffer) >= hdrlen:
            self._payload_len = struct.unpack("=i", self._recv_buffer[0:4])[0]
            self._reqid = struct.unpack("=i", self._recv_buffer[4:8])[0]
            self._subid = struct.unpack("=h", self._recv_buffer[8:hdrlen])[0]
            self._recv_buffer = self._recv_buffer[hdrlen:]

    def process_jsonheader(self):
        if len(self._recv_buffer) >= 4:
            (self._jsonheader_len, self._jsonheader, self._recv_buffer) = \
                self._json_decode(self._recv_buffer, "utf-8")

    def process_request(self):
        content_len = self._jsonheader["content-length"]
        if not len(self._recv_buffer) >= content_len:
            return

        data = self._recv_buffer[:content_len]
        self._recv_buffer = self._recv_buffer[content_len:]

        if self._jsonheader["content-type"] == "text/json":
            encoding = self._jsonheader["content-encoding"]
            (_, self._request, self._recv_buffer) = self._json_decode(data, encoding)

        elif self._jsonheader["content-type"] == "binary/array":
            shapes = self._jsonheader["array-shape"]
            dtypes = self._jsonheader["content-encoding"]
            self._request = self._array_decode(data, shapes, dtypes)

        else:
            self._request = data

        self._set_selector_events_mask("w")

    def create_response(self):
        if self._jsonheader["content-type"] == "text/json":
            response = self._create_response_json_content()

        elif self._jsonheader["content-type"] == "binary/array":
            response = self._create_response_array_content()

        else:
            response = self._create_response_json_content()

        self._send_buffer += self._create_message(**response)
        self._response_created = True
