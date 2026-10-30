"""Diffusion Transformer (DiT, Peebles & Xie, 2023), class-conditional, with classifier-free guidance.

  python models/zoo/dit.py train    [--epochs 60]
  python models/zoo/dit.py sample   [--label 7] [--cfg 3.0]
  python models/zoo/dit.py selftest          (bundled digits at 16x16, offline)

The model follows the official facebookresearch/DiT `models.py`: patchify with timm's PatchEmbed, fixed
2-D sin-cos position embeddings, timestep + label embeddings, DiT blocks with adaLN-Zero modulation
(timm Attention and Mlp), and a final adaLN layer that un-patchifies. The noise schedule and DDIM
sampling come from diffusers' schedulers. Writes one row of samples per digit class (samples.png).
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE)]
import common as C  # noqa: E402
import _data as D  # noqa: E402


def sincos_2d(dim, grid):
    """Fixed 2-D sin-cos position embedding, as in DiT/MAE."""
    def one(d, pos):
        omega = 1.0 / 10000 ** (np.arange(d // 2, dtype=np.float64) / (d / 2.0))
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], 1)
    gh, gw = np.meshgrid(np.arange(grid, dtype=np.float32), np.arange(grid, dtype=np.float32))
    return np.concatenate([one(dim // 2, gh), one(dim // 2, gw)], 1).astype(np.float32)


def make_dit(input_size=16, patch=2, in_ch=1, hidden=128, depth=6, heads=4, num_classes=10, class_dropout=0.1):
    import torch
    import torch.nn as nn
    try:
        from timm.models.vision_transformer import Attention, Mlp, PatchEmbed
    except ImportError:  # newer timm layouts
        from timm.layers import Attention, Mlp, PatchEmbed

    def modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    class TimestepEmbedder(nn.Module):
        def __init__(self, freq=256):
            super().__init__(); self.freq = freq
            self.mlp = nn.Sequential(nn.Linear(freq, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

        def forward(self, t):
            half = self.freq // 2
            f = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
            a = t[:, None].float() * f[None]
            return self.mlp(torch.cat([torch.cos(a), torch.sin(a)], -1))

    class LabelEmbedder(nn.Module):
        def __init__(self):
            super().__init__(); self.table = nn.Embedding(num_classes + 1, hidden)   # last row = "no label" (for CFG)

        def forward(self, y, train):
            if train and class_dropout > 0:
                y = torch.where(torch.rand(y.shape[0], device=y.device) < class_dropout, torch.full_like(y, num_classes), y)
            return self.table(y)

    class DiTBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6); self.attn = Attention(hidden, num_heads=heads, qkv_bias=True)
            self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
            self.mlp = Mlp(in_features=hidden, hidden_features=hidden * 4, act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))

        def forward(self, x, c):
            sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            x = x + g_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), sh_msa, sc_msa))
            return x + g_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), sh_mlp, sc_mlp))

    class FinalLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6); self.linear = nn.Linear(hidden, patch * patch * in_ch)
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))

        def forward(self, x, c):
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            return self.linear(modulate(self.norm(x), shift, scale))

    class DiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.x_embedder = PatchEmbed(input_size, patch, in_ch, hidden, bias=True)
            self.t_embedder, self.y_embedder = TimestepEmbedder(), LabelEmbedder()
            n = self.x_embedder.num_patches
            self.pos_embed = nn.Parameter(torch.from_numpy(sincos_2d(hidden, int(n ** 0.5)))[None], requires_grad=False)
            self.blocks = nn.ModuleList([DiTBlock() for _ in range(depth)]); self.final_layer = FinalLayer()
            self._init()

        def _init(self):
            def basic(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            self.apply(basic)
            w = self.x_embedder.proj.weight.data; nn.init.xavier_uniform_(w.view(w.shape[0], -1))
            nn.init.normal_(self.y_embedder.table.weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02); nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
            for b in self.blocks:  # adaLN-Zero: every block starts as the identity
                nn.init.zeros_(b.adaLN_modulation[-1].weight); nn.init.zeros_(b.adaLN_modulation[-1].bias)
            nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight); nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
            nn.init.zeros_(self.final_layer.linear.weight); nn.init.zeros_(self.final_layer.linear.bias)

        def unpatchify(self, x):
            h = w = int(x.shape[1] ** 0.5)
            x = x.reshape(x.shape[0], h, w, patch, patch, in_ch)
            return torch.einsum("nhwpqc->nchpwq", x).reshape(x.shape[0], in_ch, h * patch, w * patch)

        def forward(self, x, t, y, train=False):
            x = self.x_embedder(x) + self.pos_embed
            c = self.t_embedder(t) + self.y_embedder(y, train)
            for b in self.blocks:
                x = b(x, c)
            return self.unpatchify(self.final_layer(x, c))
    return DiT()


def sample(model, labels, size, steps, cfg, device, seed=0, num_classes=10):
    import torch
    from diffusers import DDIMScheduler
    sch = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"); sch.set_timesteps(steps)
    g = torch.Generator().manual_seed(seed); x = torch.randn(len(labels), 1, size, size, generator=g).to(device)
    y = torch.as_tensor(labels, device=device); y_null = torch.full_like(y, num_classes); model.eval()
    with torch.no_grad():
        for t in sch.timesteps:
            tt = torch.full((len(x),), int(t), device=device)
            e_c, e_u = model(x, tt, y), model(x, tt, y_null)
            x = sch.step(e_u + cfg * (e_c - e_u), t, x).prev_sample     # classifier-free guidance
    return x.clamp(-1, 1)


def train(args):
    import torch
    from diffusers import DDPMScheduler
    C.seed_all(args.seed)
    X, y = D.digits(args.size); Xt, yt = torch.as_tensor(X), torch.as_tensor(y)
    device, dname = C.get_device(args.device); C.safe_attention(dname)
    model = make_dit(args.size, args.patch, 1, args.hidden, args.depth, args.heads).to(device)
    sch = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    opt = C.make_optimizer(model.parameters(), args.lr, 0.0, dname)
    C.log("DiT · depth %d · hidden %d · patch %d · %d params · device %s" % (args.depth, args.hidden, args.patch, sum(p.numel() for p in model.parameters()), dname))
    losses = []
    for ep in range(1, args.epochs + 1):
        model.train(); perm = torch.randperm(len(Xt)); tot = 0.0
        for s in range(0, len(Xt), args.batch):
            b = perm[s:s + args.batch]; x0 = Xt[b].to(device); lab = yt[b].to(device)
            noise = torch.randn_like(x0); t = torch.randint(0, 1000, (len(x0),), device=device)
            loss = torch.nn.functional.mse_loss(model(sch.add_noise(x0, noise, t), t, lab, train=True), noise)
            opt.zero_grad(); loss.backward(); C.clip_grads(model.parameters(), 1.0, dname); opt.step(); tot += float(loss) * len(b)
        losses.append(tot / len(Xt))
        if ep == 1 or ep % max(1, args.epochs // 8) == 0:
            C.log("epoch %d  noise-prediction MSE %.4f" % (ep, losses[-1]))
    out = D.out_dir("dit"); torch.save({"state_dict": C.cpu_state(model), "cfg": vars(args) | {"device": None}}, os.path.join(out, "dit.pt"))
    labels = [d for d in range(10) for _ in range(8)]
    imgs = sample(model, labels, args.size, args.steps, args.cfg, device)
    D.save_grid(imgs, os.path.join(out, "samples.png"), nrow=8)
    from sklearn.linear_model import LogisticRegression
    judge = LogisticRegression(max_iter=2000, C=0.5).fit(X.reshape(len(X), -1), y)
    acc = float((judge.predict(imgs.cpu().numpy().reshape(len(imgs), -1)) == np.array(labels)).mean())
    C.save_json(os.path.join(out, "report.json"), {"loss_history": losses, "conditional_accuracy": acc})
    C.log("samples (one row per digit) -> %s · judge agrees with the requested digit for %.0f%%" % (os.path.join(out, "samples.png"), 100 * acc))
    return losses, acc


def sample_cmd(args):
    device, dname = C.get_device(args.device); C.safe_attention(dname); ck = C.load_ckpt(os.path.join(D.out_dir("dit"), "dit.pt")); a = ck["cfg"]
    model = make_dit(a["size"], a["patch"], 1, a["hidden"], a["depth"], a["heads"]); model.load_state_dict(ck["state_dict"]); model.to(device)
    imgs = sample(model, [args.label] * args.n, a["size"], args.steps, args.cfg, device, seed=args.seed)
    C.log("wrote", D.save_grid(imgs, os.path.join(D.out_dir("dit"), "samples_label%d.png" % args.label)))


def selftest(args):
    losses, acc = train(argparse.Namespace(size=16, patch=2, hidden=128, depth=4, heads=4, epochs=60, batch=64, lr=5e-4, steps=40, cfg=3.0, seed=0, device=args.device))
    C.check(losses[-1] < 0.6 * losses[0], "noise-prediction loss fell %.3f -> %.3f" % (losses[0], losses[-1]))
    C.check(acc > 0.2, "class conditioning works: %.0f%% of samples match their label (> 20%%, chance 10%%)" % (100 * acc))
    C.log("DIT SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--size", type=int, default=16); t.add_argument("--patch", type=int, default=2); t.add_argument("--hidden", type=int, default=192)
    t.add_argument("--depth", type=int, default=6); t.add_argument("--heads", type=int, default=6); t.add_argument("--epochs", type=int, default=60); t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-4); t.add_argument("--steps", type=int, default=50); t.add_argument("--cfg", type=float, default=3.0); t.add_argument("--seed", type=int, default=0); t.add_argument("--device")
    p = sub.add_parser("sample"); p.add_argument("--label", type=int, default=7); p.add_argument("--n", type=int, default=16); p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=3.0); p.add_argument("--seed", type=int, default=1); p.add_argument("--device")
    s = sub.add_parser("selftest"); s.add_argument("--device")
    a = ap.parse_args(); {"train": train, "sample": sample_cmd, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
