"""
Evaluate a trained checkpoint on a grid of physical error rates.

    python evaluate.py --ckpt runs/<run>/best.pt --p_test 0.02:0.12:11 --samples 500000 --out results.json
"""
import argparse
import json
import logging
import os
import sys
import time
import numpy as np
import torch

from pqecct.codes import get_code
from pqecct.noise import NoiseSampler
from pqecct.models import build_model, count_params
from train import evaluate, parse_p_list, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--p_test", type=str, required=True)
    p.add_argument("--samples", type=int, default=200000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--amp", action="store_true")
    a = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpus
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    args = argparse.Namespace(**ck["args"])
    args.amp = a.amp
    code = get_code(args.code)
    sampler = NoiseSampler(code, args.noise, args.err_types, device=device)
    model = build_model(args, code).to(device)
    model.load_state_dict(ck["model"])
    logging.info(f"{code} model={args.model} params={count_params(model):,} epoch={ck['epoch']}")
    p_list = parse_p_list(a.p_test)
    t0 = time.time()
    res = evaluate(model, sampler, p_list, a.samples, a.batch_size, args, device, "Eval")
    # latency: mean time per syndrome at batch size 1 and at a.batch_size
    with torch.no_grad():
        for bs in (1, a.batch_size):
            e_x, e_z, s_x, s_z = sampler.sample_batch(p_list[0], bs)
            for _ in range(3):
                model(s_x, s_z)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.time()
            for _ in range(10):
                model(s_x, s_z)
            if device.type == "cuda":
                torch.cuda.synchronize()
            lat = (time.time() - t) / 10 / bs
            logging.info(f"latency per syndrome at batch {bs}: {lat*1e6:.1f} us")
            res.setdefault("_meta", {})[f"latency_us_b{bs}"] = lat * 1e6
    out = dict(model=args.model, code=args.code, noise=args.noise, err_types=args.err_types, params=count_params(model),
               ckpt=a.ckpt, epoch=ck["epoch"], samples=a.samples, results=res)
    out_path = a.out or os.path.join(os.path.dirname(a.ckpt), f"eval_{a.samples}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logging.info(f"wrote {out_path} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
