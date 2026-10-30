"""Convolutional networks: LeNet-5 (LeCun et al., 1998) and torchvision AlexNet.

  python models/zoo/cnn.py train   [--dataset digits|mnist|cifar10] [--arch lenet|alexnet] [--epochs 10]
  python models/zoo/cnn.py predict --image digit.png
  python models/zoo/cnn.py selftest            (bundled 8x8 digits upsampled to 32x32, no downloads)

`mnist` and `cifar10` are downloaded by torchvision into runs/zoo/data on first use.
AlexNet expects 224x224 RGB, so inputs are resized and repeated to 3 channels; use --pretrained for
ImageNet weights (transfer learning).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE)]
import common as C  # noqa: E402
import _data as D  # noqa: E402


def lenet5(in_ch, n_classes):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(in_ch, 6, 5), nn.Tanh(), nn.AvgPool2d(2),        # 32 -> 28 -> 14
        nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2),           # 14 -> 10 -> 5
        nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.Tanh(), nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, n_classes))


def alexnet(n_classes, pretrained):
    import torch.nn as nn
    from torchvision import models
    m = models.alexnet(weights=models.AlexNet_Weights.DEFAULT if pretrained else None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    return m


def load(dataset, root):
    """Returns (X_train, y_train, X_test, y_test, channels) as float32 arrays in [0,1], 32x32."""
    if dataset == "digits":
        X, y = D.digits(32, scale="0_1"); tr, te = C.stratified_split(y, 0.2, 0)
        return X[tr], y[tr], X[te], y[te], 1
    from torchvision import datasets, transforms as T
    tf = T.Compose([T.Resize((32, 32)), T.ToTensor()])
    cls = {"mnist": datasets.MNIST, "cifar10": datasets.CIFAR10}[dataset]
    def arr(ds):
        xs, ys = zip(*[ds[i] for i in range(len(ds))]); return np.stack([x.numpy() for x in xs]), np.array(ys)
    a, b = arr(cls(root, train=True, download=True, transform=tf)), arr(cls(root, train=False, download=True, transform=tf))
    return a[0], a[1], b[0], b[1], a[0].shape[1]


def prep(x, arch):
    import torch.nn.functional as F
    if arch == "alexnet":
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
    return x


def evaluate(model, X, y, arch, device, batch=256):
    import torch
    model.eval(); pred = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            pred.append(model(prep(torch.as_tensor(X[s:s + batch], device=device), arch)).argmax(1).cpu().numpy())
    return np.concatenate(pred)


def train(args):
    import torch
    C.seed_all(args.seed)
    Xtr, ytr, Xte, yte, ch = load(args.dataset, os.path.join(D.out_dir("data")))
    n_cls = int(max(ytr.max(), yte.max()) + 1)
    device, dname = C.get_device(args.device)
    model = (lenet5(ch, n_cls) if args.arch == "lenet" else alexnet(n_cls, args.pretrained)).to(device)
    C.log("%s on %s · %d train / %d test · %d params · device %s" % (args.arch, args.dataset, len(Xtr), len(Xte), sum(p.numel() for p in model.parameters()), dname))
    opt = C.make_optimizer(model.parameters(), args.lr, 1e-4, dname); loss_fn = torch.nn.CrossEntropyLoss()
    Xt, yt = torch.as_tensor(Xtr), torch.as_tensor(ytr); best = 0.0; out = D.out_dir("cnn")
    for ep in range(1, args.epochs + 1):
        model.train(); perm = torch.randperm(len(Xt)); tot = 0.0
        for s in range(0, len(Xt), args.batch):
            b = perm[s:s + args.batch]
            x = Xt[b]
            if args.dataset != "mnist" and args.dataset != "digits":
                x = torch.where(torch.rand(len(x), 1, 1, 1) < 0.5, x.flip(3), x)  # horizontal flip for natural images only
            loss = loss_fn(model(prep(x.to(device), args.arch)), yt[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(b)
        acc = float((evaluate(model, Xte, yte, args.arch, device) == yte).mean())
        C.log("epoch %d  loss %.4f  test acc %.4f" % (ep, tot / len(Xt), acc))
        if acc >= best:
            best = acc; torch.save({"state_dict": C.cpu_state(model), "arch": args.arch, "in_ch": ch, "n_cls": n_cls}, os.path.join(out, "model.pt"))
    rep = C.cls_report(yte, evaluate(model, Xte, yte, args.arch, device), [str(i) for i in range(n_cls)]); C.print_cls_report(rep)
    C.save_json(os.path.join(out, "report.json"), {"best_test_acc": best, "final": rep, "dataset": args.dataset, "arch": args.arch})
    return best


def predict(args):
    import torch
    from PIL import Image
    device, _ = C.get_device(args.device)
    ck = C.load_ckpt(os.path.join(D.out_dir("cnn"), "model.pt"))
    model = lenet5(ck["in_ch"], ck["n_cls"]) if ck["arch"] == "lenet" else alexnet(ck["n_cls"], False)
    model.load_state_dict(ck["state_dict"]); model.to(device).eval()
    with Image.open(args.image) as im:
        a = np.asarray(im.convert("L" if ck["in_ch"] == 1 else "RGB").resize((32, 32)), np.float32) / 255.0
    x = torch.as_tensor(a[None, None] if ck["in_ch"] == 1 else a.transpose(2, 0, 1)[None], device=device)
    with torch.no_grad():
        p = torch.softmax(model(prep(x, ck["arch"])), 1)[0].cpu().numpy()
    C.log("prediction %d (%.3f) · top-3 %s" % (p.argmax(), p.max(), [(int(i), round(float(p[i]), 3)) for i in np.argsort(-p)[:3]]))
    return p


def selftest(args):
    acc = train(argparse.Namespace(dataset="digits", arch="lenet", pretrained=False, epochs=12, batch=64, lr=3e-3, seed=0, device=args.device))
    C.check(acc > 0.9, "LeNet-5 test accuracy %.3f > 0.90 on digits" % acc)
    C.log("CNN SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--dataset", choices=["digits", "mnist", "cifar10"], default="digits"); t.add_argument("--arch", choices=["lenet", "alexnet"], default="lenet")
    t.add_argument("--pretrained", action="store_true"); t.add_argument("--epochs", type=int, default=10); t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=2e-3); t.add_argument("--seed", type=int, default=0); t.add_argument("--device")
    p = sub.add_parser("predict"); p.add_argument("--image", required=True); p.add_argument("--device")
    s = sub.add_parser("selftest"); s.add_argument("--device")
    a = ap.parse_args(); {"train": train, "predict": predict, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
