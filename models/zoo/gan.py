"""DCGAN (Radford et al., 2015), following the architecture of the official pytorch/examples dcgan,
sized for 32x32 images.

  python models/zoo/gan.py train     [--epochs 40] [--nz 64]
  python models/zoo/gan.py sample    [--n 64]
  python models/zoo/gan.py selftest           (bundled digits, offline)

Generator: transposed convolutions + BatchNorm + ReLU, Tanh output. Discriminator: strided convolutions
+ BatchNorm + LeakyReLU(0.2), Sigmoid output. N(0, 0.02) weight init, Adam(lr=2e-4, betas=(0.5, 0.999)),
BCE loss, as in the paper. Quality check: a classifier trained on real digits must recognise the
generated ones with confidence (a rough proxy for "looks like a digit").
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE)]
import common as C  # noqa: E402
import _data as D  # noqa: E402


def models(nz, nc=1, ngf=64, ndf=64):
    import torch.nn as nn
    G = nn.Sequential(  # nz x 1 x 1 -> nc x 32 x 32
        nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True),        # 4x4
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True),   # 8x8
        nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True),           # 16x16
        nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh())                                         # 32x32
    Dn = nn.Sequential(
        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),                            # 16x16
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),  # 8x8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True),  # 4x4
        nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False), nn.Sigmoid(), nn.Flatten())
    def init(m):
        cn = m.__class__.__name__
        if "Conv" in cn:
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif "BatchNorm" in cn:
            nn.init.normal_(m.weight, 1.0, 0.02); nn.init.zeros_(m.bias)
    G.apply(init); Dn.apply(init)
    return G, Dn


def digit_judge(X, y):
    """Logistic-regression digit classifier on real images (flattened 32x32)."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000, C=0.5).fit(X.reshape(len(X), -1), y)


def train(args):
    import torch
    C.seed_all(args.seed)
    X, y = D.digits(32); Xt = torch.as_tensor(X)
    device, dname = C.get_device(args.device)
    G, Dn = models(args.nz); G.to(device); Dn.to(device)
    kw = {"foreach": False} if dname.startswith("directml") else {}
    oG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999), **kw); oD = torch.optim.Adam(Dn.parameters(), lr=args.lr, betas=(0.5, 0.999), **kw)
    bce = torch.nn.BCELoss(); fixed = torch.randn(64, args.nz, 1, 1, device=device); out = D.out_dir("gan")
    C.log("DCGAN · G %d / D %d params · %d real images · device %s" % (sum(p.numel() for p in G.parameters()), sum(p.numel() for p in Dn.parameters()), len(Xt), dname))
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(len(Xt)); dl = gl = dx = dgz = 0.0; n = 0
        for s in range(0, len(Xt), args.batch):
            real = Xt[perm[s:s + args.batch]].to(device); b = len(real)
            ones, zeros = torch.full((b, 1), 0.9, device=device), torch.zeros(b, 1, device=device)   # one-sided label smoothing
            fake = G(torch.randn(b, args.nz, 1, 1, device=device))
            out_r, out_f = Dn(real), Dn(fake.detach())
            lossD = bce(out_r, ones) + bce(out_f, zeros); oD.zero_grad(); lossD.backward(); oD.step()
            out_g = Dn(fake); lossG = bce(out_g, torch.ones(b, 1, device=device)); oG.zero_grad(); lossG.backward(); oG.step()
            dl += float(lossD) * b; gl += float(lossG) * b; dx += float(out_r.mean()) * b; dgz += float(out_g.mean()) * b; n += b
        if ep == 1 or ep % max(1, args.epochs // 8) == 0:
            C.log("epoch %d  loss D %.3f  G %.3f  D(x) %.2f  D(G(z)) %.2f" % (ep, dl / n, gl / n, dx / n, dgz / n))
            G.eval()
            with torch.no_grad():
                D.save_grid(G(fixed), os.path.join(out, "samples_epoch%03d.png" % ep))
            G.train()
    torch.save({"G": C.cpu_state(G), "D": C.cpu_state(Dn), "nz": args.nz}, os.path.join(out, "dcgan.pt"))
    G.eval()
    with torch.no_grad():
        gen = G(torch.randn(500, args.nz, 1, 1, device=device)).cpu().numpy()
    judge = digit_judge(X, y); conf = judge.predict_proba(gen.reshape(len(gen), -1)).max(1)
    stats = {"mean_confidence": float(conf.mean()), "share_confident": float((conf > 0.8).mean()), "pixel_std": float(gen.std()),
             "class_coverage": int(len(set(judge.predict(gen.reshape(len(gen), -1)))))}
    C.save_json(os.path.join(out, "report.json"), stats); D.save_grid(torch.as_tensor(gen[:64]), os.path.join(out, "samples_final.png"))
    C.log("generated digits: mean judge confidence %.2f · %.0f%% confident · %d of 10 classes produced · samples in %s" % (stats["mean_confidence"], 100 * stats["share_confident"], stats["class_coverage"], out))
    return stats


def sample(args):
    import torch
    ck = C.load_ckpt(os.path.join(D.out_dir("gan"), "dcgan.pt")); G, _ = models(ck["nz"]); G.load_state_dict(ck["G"]); G.eval()
    with torch.no_grad():
        p = D.save_grid(G(torch.randn(args.n, ck["nz"], 1, 1)), os.path.join(D.out_dir("gan"), "samples_new.png"))
    C.log("wrote", p)


def selftest(args):
    stats = train(argparse.Namespace(nz=64, epochs=30, batch=64, lr=2e-4, seed=0, device=args.device))
    C.check(stats["pixel_std"] > 0.2, "generator output is not collapsed (pixel std %.2f)" % stats["pixel_std"])
    C.check(stats["mean_confidence"] > 0.6 and stats["class_coverage"] >= 6, "digit judge confidence %.2f > 0.6 and %d >= 6 classes" % (stats["mean_confidence"], stats["class_coverage"]))
    C.log("GAN SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--nz", type=int, default=64); t.add_argument("--epochs", type=int, default=40); t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=2e-4); t.add_argument("--seed", type=int, default=0); t.add_argument("--device")
    p = sub.add_parser("sample"); p.add_argument("--n", type=int, default=64)
    s = sub.add_parser("selftest"); s.add_argument("--device")
    a = ap.parse_args(); {"train": train, "sample": sample, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
