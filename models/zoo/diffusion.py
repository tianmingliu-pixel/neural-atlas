"""Denoising diffusion (DDPM, Ho et al., 2020) with Hugging Face diffusers: UNet2DModel + DDPMScheduler.

  python models/zoo/diffusion.py train    [--size 32] [--epochs 40]
  python models/zoo/diffusion.py sample   [--n 64] [--steps 50]
  python models/zoo/diffusion.py selftest           (bundled digits at 16x16, offline)

Training: pick a random timestep t, add the scheduler's noise to a clean image, and train the U-Net to
predict that noise (MSE). Sampling: start from pure noise and denoise step by step (DDIM with 50 steps by
default, same trained weights). Writes samples.png and a strip showing one image being denoised.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE)]
import common as C  # noqa: E402
import _data as D  # noqa: E402


def make_unet(size):
    from diffusers import UNet2DModel
    if size <= 16:
        return UNet2DModel(sample_size=size, in_channels=1, out_channels=1, layers_per_block=1, block_out_channels=(32, 64),
                           down_block_types=("DownBlock2D", "AttnDownBlock2D"), up_block_types=("AttnUpBlock2D", "UpBlock2D"), norm_num_groups=16)
    return UNet2DModel(sample_size=size, in_channels=1, out_channels=1, layers_per_block=2, block_out_channels=(32, 64, 128),
                       down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"), up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"), norm_num_groups=16)


def sample_images(unet, n, size, steps, device, seed=0, keep_trace=False):
    import torch
    from diffusers import DDIMScheduler
    sch = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"); sch.set_timesteps(steps)
    g = torch.Generator().manual_seed(seed); x = torch.randn(n, 1, size, size, generator=g).to(device); trace = []
    unet.eval()
    with torch.no_grad():
        for i, t in enumerate(sch.timesteps):
            eps = unet(x, t).sample
            x = sch.step(eps, t, x).prev_sample
            if keep_trace and i % max(1, steps // 8) == 0:
                trace.append(x[:1].clone())
    return x.clamp(-1, 1), trace


def train(args):
    import torch
    from diffusers import DDPMScheduler
    C.seed_all(args.seed)
    X, _ = D.digits(args.size); Xt = torch.as_tensor(X)
    device, dname = C.get_device(args.device)
    unet = C.safe_attention(dname, make_unet(args.size)).to(device); sch = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    opt = C.make_optimizer(unet.parameters(), args.lr, 0.0, dname)
    C.log("UNet2DModel · %d params · %d images %dx%d · device %s" % (sum(p.numel() for p in unet.parameters()), len(Xt), args.size, args.size, dname))
    losses = []
    for ep in range(1, args.epochs + 1):
        unet.train(); perm = torch.randperm(len(Xt)); tot = 0.0
        for s in range(0, len(Xt), args.batch):
            x0 = Xt[perm[s:s + args.batch]].to(device)
            noise = torch.randn_like(x0); t = torch.randint(0, 1000, (len(x0),), device=device)
            loss = torch.nn.functional.mse_loss(unet(sch.add_noise(x0, noise, t), t).sample, noise)
            opt.zero_grad(); loss.backward(); C.clip_grads(unet.parameters(), 1.0, dname); opt.step(); tot += float(loss) * len(x0)
        losses.append(tot / len(Xt))
        if ep == 1 or ep % max(1, args.epochs // 8) == 0:
            C.log("epoch %d  noise-prediction MSE %.4f" % (ep, losses[-1]))
    out = D.out_dir("diffusion"); torch.save({"state_dict": C.cpu_state(unet), "size": args.size}, os.path.join(out, "unet.pt"))
    imgs, trace = sample_images(unet, 64, args.size, args.steps, device, keep_trace=True)
    D.save_grid(imgs, os.path.join(out, "samples.png")); D.save_grid(torch.cat(trace + [imgs[:1]]), os.path.join(out, "denoising.png"), nrow=len(trace) + 1)
    C.save_json(os.path.join(out, "report.json"), {"loss_history": losses})
    C.log("samples -> %s" % os.path.join(out, "samples.png"))
    return losses, imgs.cpu().numpy()


def sample(args):
    import torch
    device, dname = C.get_device(args.device); ck = C.load_ckpt(os.path.join(D.out_dir("diffusion"), "unet.pt"))
    unet = C.safe_attention(dname, make_unet(ck["size"])); unet.load_state_dict(ck["state_dict"]); unet.to(device)
    imgs, _ = sample_images(unet, args.n, ck["size"], args.steps, device, seed=args.seed)
    C.log("wrote", D.save_grid(imgs, os.path.join(D.out_dir("diffusion"), "samples_new.png")))


def selftest(args):
    losses, imgs = train(argparse.Namespace(size=16, epochs=20, batch=64, lr=1e-3, steps=40, seed=0, device=args.device))
    C.check(losses[-1] < 0.6 * losses[0], "noise-prediction loss fell %.3f -> %.3f" % (losses[0], losses[-1]))
    C.check(np.isfinite(imgs).all() and imgs.std() > 0.2, "sampling produces finite, non-constant images (std %.2f)" % imgs.std())
    C.log("DIFFUSION SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--size", type=int, default=32); t.add_argument("--epochs", type=int, default=40); t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=5e-4); t.add_argument("--steps", type=int, default=50); t.add_argument("--seed", type=int, default=0); t.add_argument("--device")
    p = sub.add_parser("sample"); p.add_argument("--n", type=int, default=64); p.add_argument("--steps", type=int, default=50); p.add_argument("--seed", type=int, default=1); p.add_argument("--device")
    s = sub.add_parser("selftest"); s.add_argument("--device")
    a = ap.parse_args(); {"train": train, "sample": sample, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
