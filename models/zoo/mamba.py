"""Mamba selective state-space model (Gu & Dao, 2023) on the selective-copying task.

  python models/zoo/mamba.py train      [--steps 3000] [--impl auto|official|reference]
  python models/zoo/mamba.py selftest

Implementations:
  official  : `mamba_ssm.Mamba` (pip install mamba-ssm) — requires an NVIDIA GPU with CUDA on Linux.
  reference : a line-by-line PyTorch port of the Mamba block (in_proj -> causal depthwise conv ->
              input-dependent Δ, B, C -> selective scan -> gated output), after the paper and the
              "mamba-minimal" reference. Runs anywhere (CPU, DirectML), with a sequential scan, so it is slower.
`auto` picks official when available, otherwise reference.

Selective copying: a sequence holds K data tokens at random positions among blanks; after a marker the
model must output the K tokens in order. Solving it needs input-dependent selection — what Mamba adds over
earlier linear-time state-space models.
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE)]
import common as C  # noqa: E402
import _data as D  # noqa: E402

N_DATA, BLANK, MARK = 8, 0, 9       # data tokens are 1..8
VOCAB = 10


def batch(n, L=16, K=4, rng=None):
    rng = rng or np.random.default_rng()
    x = np.zeros((n, L + 1 + K), np.int64); y = np.full((n, L + 1 + K), -100, np.int64)
    for i in range(n):
        pos = np.sort(rng.choice(L, K, replace=False)); toks = rng.integers(1, N_DATA + 1, K)
        x[i, pos] = toks; x[i, L] = MARK; y[i, L + 1:] = toks
    return x, y


def reference_block(d_model, d_state=16, d_conv=4, expand=2):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    d_inner, dt_rank = expand * d_model, math.ceil(d_model / 16)

    class MambaBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
            self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
            self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
            self.dt_proj = nn.Linear(dt_rank, d_inner)
            # official Mamba initialisation of the step size: dt ~ log-uniform[1e-3, 1e-1], bias = softplus^-1(dt)
            dt = torch.exp(torch.rand(d_inner) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)).clamp(min=1e-4)
            with torch.no_grad():
                self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            nn.init.uniform_(self.dt_proj.weight, -dt_rank ** -0.5, dt_rank ** -0.5)
            self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)))
            self.D = nn.Parameter(torch.ones(d_inner))
            self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        def forward(self, x):                                     # x: [B, L, d_model]
            L = x.shape[1]
            u, res = self.in_proj(x).chunk(2, -1)
            u = F.silu(self.conv1d(u.transpose(1, 2))[:, :, :L].transpose(1, 2))   # causal depthwise conv
            delta, Bm, Cm = self.x_proj(u).split([dt_rank, d_state, d_state], -1)
            delta = F.softplus(self.dt_proj(delta))               # input-dependent step size Δ
            A = -torch.exp(self.A_log)                            # [d_inner, d_state]
            dA = torch.exp(torch.einsum("bld,dn->bldn", delta, A))
            dBu = torch.einsum("bld,bln,bld->bldn", delta, Bm, u)
            h = torch.zeros(x.shape[0], d_inner, d_state, device=x.device); ys = []
            for t in range(L):                                    # selective scan
                h = dA[:, t] * h + dBu[:, t]
                ys.append(torch.einsum("bdn,bn->bd", h, Cm[:, t]))
            y = torch.stack(ys, 1) + u * self.D
            return self.out_proj(y * F.silu(res))
    return MambaBlock()


def make_model(d_model, layers, impl):
    import torch
    import torch.nn as nn

    class RMSNorm(nn.Module):
        def __init__(self, d, eps=1e-5):
            super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w

    def mixer():
        if impl == "official":
            from mamba_ssm import Mamba
            return Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        return reference_block(d_model)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(VOCAB, d_model)
            self.norms = nn.ModuleList([RMSNorm(d_model) for _ in range(layers)]); self.mixers = nn.ModuleList([mixer() for _ in range(layers)])
            self.norm_f = RMSNorm(d_model); self.head = nn.Linear(d_model, VOCAB)

        def forward(self, x):
            h = self.emb(x)
            for n, m in zip(self.norms, self.mixers):
                h = h + m(n(h))
            return self.head(self.norm_f(h))
    return Net()


def pick_impl(impl):
    if impl in ("auto", "official"):
        try:
            import torch
            import mamba_ssm  # noqa: F401
            if torch.cuda.is_available():
                return "official"
        except Exception:
            pass
        if impl == "official":
            C.log("mamba-ssm (or CUDA) is not available here; using the reference implementation")
    return "reference"


def accuracy(model, device, rng, n=512):
    import torch
    x, y = batch(n, rng=rng); model.eval()
    with torch.no_grad():
        p = model(torch.as_tensor(x, device=device)).argmax(-1).cpu().numpy()
    m = y != -100
    return float((p[m] == y[m]).mean())


def train(args):
    import torch
    C.seed_all(args.seed); rng = np.random.default_rng(args.seed)
    impl = pick_impl(args.impl)
    device, dname = (torch.device("cuda"), "cuda") if impl == "official" else C.get_device(args.device)
    model = make_model(args.d_model, args.layers, impl).to(device)
    opt = C.make_optimizer(model.parameters(), args.lr, 0.01, dname); loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    C.log("Mamba (%s) · %d layers · d_model %d · %d params · device %s" % (impl, args.layers, args.d_model, sum(p.numel() for p in model.parameters()), dname))
    for step in range(1, args.steps + 1):
        model.train(); x, y = batch(args.batch, rng=rng)
        logits = model(torch.as_tensor(x, device=device))
        loss = loss_fn(logits.reshape(-1, VOCAB), torch.as_tensor(y, device=device).reshape(-1))
        opt.zero_grad(); loss.backward(); C.clip_grads(model.parameters(), 1.0, dname); opt.step(); sched.step()
        if step % max(1, args.steps // 10) == 0 or step == 1:
            C.log("step %d  loss %.4f  copy accuracy %.3f" % (step, float(loss), accuracy(model, device, rng)))
    acc = accuracy(model, device, rng, 2000)
    out = D.out_dir("mamba"); torch.save({"state_dict": C.cpu_state(model), "impl": impl, "d_model": args.d_model, "layers": args.layers}, os.path.join(out, "model.pt"))
    C.save_json(os.path.join(out, "report.json"), {"impl": impl, "selective_copy_accuracy": acc, "chance": 1 / N_DATA})
    C.log("selective-copy token accuracy %.3f (chance %.3f)" % (acc, 1 / N_DATA))
    return acc


def selftest(args):
    acc = train(argparse.Namespace(impl="reference", d_model=64, layers=2, steps=3000, batch=64, lr=3e-3, seed=0, device=args.device))
    C.check(acc > 0.4, "selective-copy accuracy %.3f > 0.40 (chance 0.125)" % acc)
    C.log("MAMBA SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--impl", choices=["auto", "official", "reference"], default="auto"); t.add_argument("--d-model", dest="d_model", type=int, default=64)
    t.add_argument("--layers", type=int, default=2); t.add_argument("--steps", type=int, default=3000); t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-3); t.add_argument("--seed", type=int, default=0); t.add_argument("--device")
    s = sub.add_parser("selftest"); s.add_argument("--device")
    a = ap.parse_args(); {"train": train, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
