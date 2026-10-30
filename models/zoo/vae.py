"""Variational autoencoder (Kingma & Welling, 2013) with diffusers' AutoencoderKL.

  python models/zoo/vae.py train     [--size 32] [--epochs 30] [--beta 1e-3]
  python models/zoo/vae.py selftest            (bundled digits, offline)

Loss = reconstruction MSE + beta * KL(q(z|x) || N(0, I)). Saves reconstructions, samples drawn from
the prior and a latent-space interpolation between two digits (recon.png, samples.png, interp.png).
AutoencoderKL is the same class Stable Diffusion uses for its latent space, configured small here.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE)]
import common as C  # noqa: E402
import _data as D  # noqa: E402


def make_vae(latent=4):
    from diffusers import AutoencoderKL
    return AutoencoderKL(in_channels=1, out_channels=1, down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"), up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
                         block_out_channels=(32, 64), layers_per_block=1, latent_channels=latent, norm_num_groups=32)


def train(args):
    import torch
    C.seed_all(args.seed)
    X, y = D.digits(args.size); tr, va = C.stratified_split(y, 0.1, 0)
    device, dname = C.get_device(args.device)
    vae = C.safe_attention(dname, make_vae(args.latent)).to(device); opt = C.make_optimizer(vae.parameters(), args.lr, 0.0, dname)
    Xt = torch.as_tensor(X[tr]); Xv = torch.as_tensor(X[va]).to(device); first = None
    C.log("AutoencoderKL · %d params · %d train images %dx%d · device %s" % (sum(p.numel() for p in vae.parameters()), len(Xt), args.size, args.size, dname))
    def val_mse():
        vae.eval()
        with torch.no_grad():
            return float(((vae.decode(vae.encode(Xv).latent_dist.mode()).sample - Xv) ** 2).mean())
    first = val_mse()
    for ep in range(1, args.epochs + 1):
        vae.train(); perm = torch.randperm(len(Xt)); rec_t = kl_t = 0.0
        for s in range(0, len(Xt), args.batch):
            x = Xt[perm[s:s + args.batch]].to(device)
            post = vae.encode(x).latent_dist; z = post.sample()
            rec = ((vae.decode(z).sample - x) ** 2).mean(); kl = post.kl().mean() / x[0].numel()
            loss = rec + args.beta * kl
            opt.zero_grad(); loss.backward(); opt.step(); rec_t += float(rec) * len(x); kl_t += float(kl) * len(x)
        if ep == 1 or ep % max(1, args.epochs // 6) == 0:
            C.log("epoch %d  recon %.4f  KL/pixel %.4f  val recon %.4f" % (ep, rec_t / len(Xt), kl_t / len(Xt), val_mse()))
    last = val_mse(); out = D.out_dir("vae"); vae.eval()
    with torch.no_grad():
        post = vae.encode(Xv[:16]).latent_dist
        D.save_grid(torch.cat([Xv[:16], vae.decode(post.mode()).sample]), os.path.join(out, "recon.png"), nrow=16)
        zshape = post.mode().shape[1:]
        D.save_grid(vae.decode(torch.randn(32, *zshape, device=device)).sample, os.path.join(out, "samples.png"))
        za, zb = post.mode()[0:1], post.mode()[1:2]; alphas = torch.linspace(0, 1, 10, device=device)[:, None, None, None]
        D.save_grid(vae.decode(za * (1 - alphas) + zb * alphas).sample, os.path.join(out, "interp.png"), nrow=10)
    torch.save({"state_dict": C.cpu_state(vae), "latent": args.latent, "size": args.size}, os.path.join(out, "vae.pt"))
    C.save_json(os.path.join(out, "report.json"), {"val_recon_mse_start": first, "val_recon_mse_end": last, "beta": args.beta})
    C.log("val reconstruction MSE %.4f -> %.4f · images in %s" % (first, last, out))
    return first, last


def selftest(args):
    first, last = train(argparse.Namespace(size=16, latent=4, epochs=15, batch=64, lr=2e-3, beta=1e-3, seed=0, device=args.device))
    C.check(last < first * 0.3 and last < 0.08, "validation reconstruction MSE %.4f -> %.4f" % (first, last))
    C.log("VAE SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--size", type=int, default=32); t.add_argument("--latent", type=int, default=4); t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=64); t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--beta", type=float, default=1e-3); t.add_argument("--seed", type=int, default=0); t.add_argument("--device")
    s = sub.add_parser("selftest"); s.add_argument("--device")
    a = ap.parse_args(); {"train": train, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
