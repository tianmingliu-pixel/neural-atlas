"""Shared helpers for the Neural Forge training scripts.

Device selection order: CUDA (also covers AMD ROCm builds on Linux) -> DirectML (AMD/Intel on Windows,
`pip install torch-directml`) -> Apple MPS -> CPU. Override with --device cpu|cuda|directml|mps.
"""
import datetime
import json
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:  # never crash on consoles whose code page lacks a character we print
    sys.stdout.reconfigure(errors="replace"); sys.stderr.reconfigure(errors="replace")
except Exception:
    pass
RUNS = os.path.join(ROOT, "runs")


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def get_device(prefer=None):
    """Return (device, name). `prefer` may be None/'auto', 'cpu', 'cuda', 'directml' or 'mps'."""
    import torch
    prefer = (prefer or "auto").lower()
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda"), "cuda (" + torch.cuda.get_device_name(0) + ")"
    if prefer in ("auto", "directml"):
        try:
            import torch_directml  # type: ignore
            dev = torch_directml.device()
            try:
                name = torch_directml.device_name(0)
            except Exception:
                name = "GPU"
            return dev, "directml (" + name + ")"
        except Exception:
            if prefer == "directml":
                log("torch-directml is not installed or not usable; falling back to CPU")
    if prefer in ("auto", "mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    if prefer not in ("auto", "cpu"):
        log("requested device '%s' is not available; using CPU" % prefer)
    return torch.device("cpu"), "cpu"


def make_optimizer(params, lr, weight_decay, device_name):
    """AdamW; the fused/foreach kernels are disabled on DirectML, which does not support them."""
    import torch
    kw = {"lr": lr, "weight_decay": weight_decay}
    if device_name.startswith("directml"):
        kw["foreach"] = False
    return torch.optim.AdamW(params, **kw)


def new_run_dir(task, name=None):
    """Self-test runs go to runs/_selftest and never become the 'latest' run of a task."""
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if name and name.startswith("selftest"):
        d = os.path.join(RUNS, "_selftest", task, name)
        os.makedirs(d, exist_ok=True)
        return d
    d = os.path.join(RUNS, task, name or stamp)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(RUNS, task, "latest.txt"), "w", encoding="utf-8") as f:
        f.write(d)
    return d


def cpu_state(model):
    """State dict copied to CPU: DirectML tensors cannot be pickled or written to safetensors directly."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_ckpt(path):
    """Load a checkpoint written by these scripts (tensors, numbers, strings and lists only)."""
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # very old torch without weights_only
        return torch.load(path, map_location="cpu")


def safe_attention(device_name, model=None):
    """On DirectML, avoid fused scaled-dot-product attention kernels and nn.Transformer fast paths.
    Call once before building timm models, and again with `model` for diffusers models."""
    if not device_name.startswith("directml"):
        return model
    import torch
    try:
        torch.backends.mha.set_fastpath_enabled(False)
    except Exception:
        pass
    try:
        from timm.layers import set_fused_attn
        set_fused_attn(False)
    except Exception:
        pass
    if model is not None:
        try:
            from diffusers.models.attention_processor import Attention, AttnProcessor
            for m in model.modules():
                if isinstance(m, Attention):
                    m.set_processor(AttnProcessor())
        except Exception:
            pass
    return model


def clip_grads(params, max_norm, device_name):
    import torch
    kw = {"foreach": False} if device_name.startswith("directml") else {}
    try:
        return torch.nn.utils.clip_grad_norm_(params, max_norm, **kw)
    except TypeError:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)


def resolve_run(task, run):
    """`run` may be a directory, or None/'latest' for the most recent run of `task`."""
    if run and run != "latest":
        return run
    p = os.path.join(RUNS, task, "latest.txt")
    if not os.path.exists(p):
        sys.exit("No trained run for '%s' yet. Train first." % task)
    with open(p, encoding="utf-8") as f:
        return f.read().strip()


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def stratified_split(labels, val_frac=0.2, seed=42):
    """Indices for train/val; stratified when every class has at least 2 samples."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(labels))
    _, counts = np.unique(labels, return_counts=True)
    strat = labels if counts.min() >= 2 else None
    tr, va = train_test_split(idx, test_size=val_frac, random_state=seed, stratify=strat)
    return np.sort(tr), np.sort(va)


def cls_report(y_true, y_pred, class_names):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {class_names[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i in labels},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": list(class_names),
    }


def print_cls_report(rep):
    log("accuracy %.3f · macro F1 %.3f" % (rep["accuracy"], rep["macro_f1"]))
    names = rep["labels"]; w = max(8, max(len(n) for n in names))
    print("  " + "true\\pred".ljust(w) + " ".join(n[:7].rjust(7) for n in names))
    for i, row in enumerate(rep["confusion"]):
        print("  " + names[i][:w].ljust(w) + " ".join(str(v).rjust(7) for v in row))


class EarlyStopper:
    def __init__(self, patience=5, mode="max", min_delta=1e-4):
        self.patience, self.mode, self.min_delta = patience, mode, min_delta
        self.best, self.bad = None, 0

    def step(self, value):
        """Return True when `value` is a new best."""
        better = self.best is None or (value > self.best + self.min_delta if self.mode == "max" else value < self.best - self.min_delta)
        if better:
            self.best, self.bad = value, 0
        else:
            self.bad += 1
        return better

    @property
    def stop(self):
        return self.bad >= self.patience


def class_weights(y, n_classes):
    counts = np.bincount(np.asarray(y), minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1
    return (len(y) / (n_classes * counts)).astype(np.float32)


def check(cond, msg):
    """Self-test assertion with a readable message."""
    if cond:
        log("PASS", msg)
    else:
        log("FAIL", msg)
        sys.exit(1)
