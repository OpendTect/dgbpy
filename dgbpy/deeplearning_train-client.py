#!/usr/bin/env python3
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
#

import argparse
import numpy as np
from os import path
import psutil
import selectors
import socket
import sys
import time
import traceback

from odpy.common import initLogging, redirect_stdout, log_msg, std_msg
from odpy import oscommand
import dgbpy.deeplearning_train_clientlib as trainlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Client for deep-learning training server"
    )
    parser.add_argument("modelfile", type=argparse.FileType("r"),
                        help="Input pretrained model file")
    parser.add_argument("--address", dest="addr", type=str, default="localhost")
    parser.add_argument("--port", dest="port", type=int, default=None)
    parser.add_argument("--gamma", dest="gamma", type=float, default=0.5)
    parser.add_argument("--tr-steps", dest="tr_steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Augmented crops per optimizer update (default: 10).")
    parser.add_argument("--crop-size", type=int, default=256,
                        help="Native-resolution training crop size (default: 256). "
                             "Must be divisible by 16; input dimensions must be at least this size.")
    parser.add_argument("--log-every-n-steps", dest="log_every_n_steps", type=int, default=0,
                        help="When >0, log training loss every N optimization steps.")
    parser.add_argument("--trainer-id", dest="trainer_id", type=str,
                        default="faultnet_transfer")
    parser.add_argument("--output-model-name", dest="output_model_name",
                        type=str, default="Fault-Net-Retrained")
    parser.add_argument("--seismic-npy", dest="seismic_npy", type=str, default="",
                        help="Path to .npy file containing seismic training array.")
    parser.add_argument("--label-npy", dest="label_npy", type=str, default="",
                        help="Path to .npy file containing label training array.")
    parser.add_argument("--log", dest="logfile", nargs="?",
                        type=argparse.FileType("w"), default=sys.stdout)
    parser.add_argument("--syslog", dest="sysout", nargs="?",
                        type=argparse.FileType("w"), default=sys.stdout)
    parser.add_argument("--server-log", dest="servlogfile", type=str, default="stdout")
    parser.add_argument("--server-syslog", dest="servsysout", type=str, default="stdout")
    return parser.parse_args()


def create_request(action, value=None):
    if value is None:
        return {"type": "text/json", "encoding": "utf-8", "content": {"action": action}}
    if action == "data":
        return {
            "type": "binary/array",
            "encoding": [arr.dtype.name for arr in value],
            "content": value,
        }
    return {
        "type": "text/json",
        "encoding": "utf-8",
        "content": {"action": action, "value": value},
    }


def wait_for_port(sockfam, addr, timeout=20.0):
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket(sockfam, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            if s.connect_ex(addr) == 0:
                return True
        time.sleep(0.05)
    return False


def req_connection(serverproc, sel, sockfam, addr, request):
    sock = socket.socket(sockfam, socket.SOCK_STREAM)
    sock.setblocking(False)
    sock.connect_ex(addr)
    message = trainlib.Message(sel, sock, addr, request)
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    sel.register(sock, events, data=message)
    servererror = None
    serverresp = None
    ret = True
    while servererror is None:
        if serverproc.poll() is not None:
            std_msg("Train server exited or crashed")
            break
        events = sel.select(timeout=1)
        for key, mask in events:
            msg = key.data
            done = False
            try:
                msg.process_events(mask)
            except Exception:
                std_msg("client exception", traceback.format_exc())
                msg.close()
                ret = False
            finally:
                servererror = msg._serverexception
                serverresp = msg._response
                if mask & selectors.EVENT_READ and msg._killreqhandled:
                    done = True
            if done:
                break
        if not sel.get_map():
            break
    return (ret if servererror is None else False), serverresp


def get_server_cmd(args, port):
    servscriptfp = path.join(path.dirname(__file__), "deeplearning_train-server.py")
    cmd = [
        oscommand.getPythonExecNm(),
        servscriptfp,
        args["modelfile"].name,
        "--address", str(args["addr"]),
        "--port", str(port),
        "--ppid", str(psutil.Process().pid),
    ]
    if args["servlogfile"] != "stdout":
        cmd.extend(["--log", args["servlogfile"]])
    if args["servsysout"] != "stdout":
        cmd.extend(["--syslog", args["servsysout"]])
    return cmd


def _load_array_from_npy(fnm):
    loaded = np.load(fnm)
    if isinstance(loaded, np.ndarray):
        return loaded
    if hasattr(loaded, "files") and loaded.files:
        return loaded[loaded.files[0]]
    raise ValueError(f"File '{fnm}' does not contain a loadable array.")


def main() -> int:
    args = vars(parse_args())
    initLogging(args)
    redirect_stdout()

    rng = np.random.default_rng()
    if args["seismic_npy"]:
        seismic = _load_array_from_npy(args["seismic_npy"]).astype(np.float32, copy=False)
    else:
        seismic = rng.standard_normal(size=(1, 1, 64, 64, 64), dtype=np.float32)

    if args["label_npy"]:
        labels = _load_array_from_npy(args["label_npy"]).astype(np.float32, copy=False)
    else:
        labels = np.full(seismic.shape, -1.0, dtype=np.float32)

    if seismic.shape != labels.shape:
        std_msg(f"Shape mismatch between seismic {seismic.shape} and labels {labels.shape}")
        return 1

    host, port = args["addr"], args["port"]
    if port is None:
        port = rng.integers(60000, high=65432, dtype=np.uint16)
    if host == "LOCAL":
        sockfam = socket.AF_UNIX
        addr = str(port)
    else:
        sockfam = socket.AF_INET
        addr = (host, port)

    servercmd = get_server_cmd(args, port)
    serverproc = oscommand.execCommand(servercmd, background=True)
    if not wait_for_port(sockfam, addr):
        std_msg("Train server never opened its port")
        serverproc.terminate()
        serverproc.wait()
        return 1

    ex_code = 0
    sel = selectors.DefaultSelector()
    try:
        ok, _ = req_connection(serverproc, sel, sockfam, addr, create_request("status"))
        if ok:
            pars = {
                "trainer_id": args["trainer_id"],
                "trainer_params": {
                    "gamma": args["gamma"],
                    "tr_steps": args["tr_steps"],
                    "crop_size": args["crop_size"],
                    "batch_size": args["batch_size"],
                    "log_every_n_steps": args["log_every_n_steps"],
                },
                "output_model_name": args["output_model_name"],
            }
            ok, _ = req_connection(serverproc, sel, sockfam, addr, create_request("parameters", pars))
            if ok:
                ok, _ = req_connection(serverproc, sel, sockfam, addr,
                                       create_request("data", [seismic]))
                if not ok:
                    ex_code = 1
                if ex_code == 0:
                    ok, _ = req_connection(serverproc, sel, sockfam, addr,
                                           create_request("data", [labels]))
                    if not ok:
                        ex_code = 1
                if ex_code == 0:
                    ok, finresp = req_connection(serverproc, sel, sockfam, addr,
                                                 create_request("finalize"))
                    if not ok:
                        ex_code = 1
                    elif finresp and "metrics" in finresp:
                        log_msg(f"Finalize metrics: {finresp['metrics']}")

                if ex_code == 0:
                    ok, savresp = req_connection(
                        serverproc, sel, sockfam, addr,
                        create_request("save", {"output_model_name": args["output_model_name"]})
                    )
                    if not ok:
                        ex_code = 1
                    elif savresp and "output_file" in savresp:
                        log_msg(f"Saved model: {savresp['output_file']}")
                ok, _ = req_connection(serverproc, sel, sockfam, addr, create_request("kill"))
                if not ok:
                    ex_code = 1
            else:
                ex_code = 1
        else:
            ex_code = 1
    except Exception:
        std_msg("Unexpected error:", traceback.format_exc())
        ex_code = 1
    finally:
        sel.close()
        time.sleep(1)
        if serverproc and serverproc.poll() is None:
            try:
                oscommand.kill( serverproc )
            except Exception as exc:
                std_msg(f"Warning: failed to kill train server process cleanly: {exc!r}")
    if ex_code == 0:
        log_msg("Train client completed successfully")
    return ex_code


if __name__ == "__main__":
    sys.exit(main())
