"""Image classification by transfer learning with torchvision ResNet (ImageNet weights) + Grad-CAM.

  python models/image_classify.py train   --data path\\to\\images   (one sub-folder per class)
  python models/image_classify.py predict --image scan.png [--run latest]   -> label, probabilities, Grad-CAM overlay PNG
  python models/image_classify.py selftest   (offline: generated images, no weight download)

Grayscale scans (X-rays) are converted to 3 channels. Training runs in two phases: first only the new
classification head (backbone frozen), then the whole network at a lower learning rate. Weighted loss for
imbalanced classes, early stopping on validation macro F1.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

TASK = "image"
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def list_images(root):
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    items = []
    for ci, c in enumerate(classes):
        for dp, _, fs in os.walk(os.path.join(root, c)):
            items += [(os.path.join(dp, f), ci) for f in sorted(fs) if f.lower().endswith(EXTS)]
    return classes, items


def to_rgb(im):
    return im.convert("RGB")


def transforms(size, train):
    from torchvision import transforms as T
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train:
        return T.Compose([T.Lambda(to_rgb), T.RandomResizedCrop(size, scale=(0.75, 1.0)), T.RandomHorizontalFlip(),
                          T.RandomRotation(8), T.ColorJitter(0.15, 0.15), T.ToTensor(), norm])
    return T.Compose([T.Lambda(to_rgb), T.Resize(int(size * 1.14)), T.CenterCrop(size), T.ToTensor(), norm])


class Files:
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        p, y = self.items[i]
        with Image.open(p) as im:
            return self.tf(im.copy()), y


def make_model(arch, n_classes, pretrained):
    import torch.nn as nn
    from torchvision import models
    weights = {"resnet18": models.ResNet18_Weights.DEFAULT, "resnet50": models.ResNet50_Weights.DEFAULT}[arch] if pretrained else None
    m = getattr(models, arch)(weights=weights)
    m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.fc.in_features, n_classes))
    return m


def run_eval(model, loader, device):
    import torch
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(model(x.to(device)).argmax(1).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train(args):
    import torch
    from torch.utils.data import DataLoader
    C.seed_all(args.seed)
    classes, items = list_images(args.data)
    if len(classes) < 2:
        sys.exit("Need at least 2 class sub-folders in %s" % args.data)
    y = np.array([c for _, c in items]); tr, va = C.stratified_split(y, args.val, args.seed)
    device, dname = C.get_device(args.device)
    C.log("%d classes %s · %d train / %d val · %s · device %s" % (len(classes), classes, len(tr), len(va), args.arch, dname))
    dl_tr = DataLoader(Files([items[i] for i in tr], transforms(args.size, True)), batch_size=args.batch, shuffle=True, num_workers=args.workers)
    dl_va = DataLoader(Files([items[i] for i in va], transforms(args.size, False)), batch_size=args.batch, num_workers=args.workers)
    model = make_model(args.arch, len(classes), not args.no_pretrained).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.as_tensor(C.class_weights(y[tr], len(classes)), device=device))
    out = C.new_run_dir(TASK, args.name)
    stopper, best, hist = C.EarlyStopper(args.patience, "max"), None, []
    for phase, (epochs, lr, freeze) in enumerate([(args.head_epochs, args.lr * 10, True), (args.epochs, args.lr, False)]):
        for n, p in model.named_parameters():
            p.requires_grad = (not freeze) or n.startswith("fc.")
        opt = C.make_optimizer([p for p in model.parameters() if p.requires_grad], lr, args.wd, dname)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, epochs))
        for ep in range(1, epochs + 1):
            model.train(); tot = 0.0
            for x, yb in dl_tr:
                loss = loss_fn(model(x.to(device)), yb.to(device))
                opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(yb)
            sched.step()
            yt, yp = run_eval(model, dl_va, device); rep = C.cls_report(yt, yp, classes)
            hist.append({"phase": "head" if freeze else "full", "epoch": ep, "loss": tot / len(tr), "val_acc": rep["accuracy"], "val_macro_f1": rep["macro_f1"]})
            C.log("%s epoch %d  loss %.4f  val acc %.3f  F1 %.3f" % ("head" if freeze else "full", ep, tot / len(tr), rep["accuracy"], rep["macro_f1"]))
            if stopper.step(rep["macro_f1"]):
                best = rep; torch.save({"state_dict": C.cpu_state(model), "arch": args.arch, "classes": classes, "size": args.size}, os.path.join(out, "model.pt"))
            if not freeze and stopper.stop:
                C.log("early stop"); break
    C.print_cls_report(best)
    C.save_json(os.path.join(out, "report.json"), {"classes": classes, "validation": best, "history": hist, "device": dname, "arch": args.arch, "pretrained": not args.no_pretrained})
    C.log("saved to", out)
    return out, best


def gradcam(model, x, target):
    """Grad-CAM on the last ResNet block; returns an HxW map in [0,1] at input resolution."""
    import torch
    import torch.nn.functional as F
    feats = {}
    def fwd(m, i, o):
        feats["a"] = o
        o.register_hook(lambda g: feats.__setitem__("g", g))
    h1 = model.layer4.register_forward_hook(fwd)
    with torch.enable_grad():
        model.zero_grad(); out = model(x); out[0, target].backward()
    h1.remove()
    w = feats["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * feats["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min(); cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy(), out.detach()


def predict(args):
    import torch
    from PIL import Image
    run = C.resolve_run(TASK, args.run)
    device, dname = C.get_device(args.device)
    ck = C.load_ckpt(os.path.join(run, "model.pt"))
    model = make_model(ck["arch"], len(ck["classes"]), False); model.load_state_dict(ck["state_dict"]); model.to(device).eval()
    with Image.open(args.image) as im:
        src = im.convert("RGB")
    x = transforms(ck["size"], False)(src).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0].cpu().numpy()
    k = int(p.argmax())
    cam, _ = gradcam(model, x, k)
    from torchvision import transforms as T
    base = T.CenterCrop(ck["size"])(T.Resize(int(ck["size"] * 1.14))(src)); arr = np.asarray(base).astype(np.float32)  # same geometry as the model input
    heat = np.stack([cam * 255, cam * 120, np.zeros_like(cam)], -1)
    over = Image.fromarray(np.clip(arr * 0.55 + heat * 0.6, 0, 255).astype(np.uint8))
    outp = args.out or os.path.splitext(args.image)[0] + "_gradcam.png"; over.save(outp)
    res = {"image": args.image, "label": ck["classes"][k], "confidence": round(float(p[k]), 4), "probs": {c: round(float(v), 4) for c, v in zip(ck["classes"], p)}, "gradcam": outp}
    C.log("%s -> %s (%.3f) · Grad-CAM saved to %s" % (args.image, res["label"], res["confidence"], outp))
    return res


def synthetic(root, per_class=70, size=64, seed=0):
    """X-ray-like patches: one class has a round bright spot, the other does not."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    for cls in ("clear", "spot"):
        os.makedirs(os.path.join(root, cls), exist_ok=True)
        for n in range(per_class):
            yy, xx = np.mgrid[0:size, 0:size]
            v = 0.35 + 0.18 * np.sin(yy * (0.35 + rng.random() * .15) / (size / 32) + rng.random() * 6) + rng.normal(0, .07, (size, size))
            if cls == "spot":
                cx, cy, r = rng.integers(14, size - 14, 2).tolist() + [2.5 + rng.random() * 3]
                v = v + (0.55 + rng.random() * .3) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))
            Image.fromarray((np.clip(v, 0, 1) * 255).astype(np.uint8), "L").save(os.path.join(root, cls, "%03d.png" % n))


def selftest(args):
    import tempfile
    tmp = tempfile.mkdtemp(prefix="nf_img_"); data = os.path.join(tmp, "data"); synthetic(data)
    classes, items = list_images(data)
    C.check(classes == ["clear", "spot"] and len(items) == 140, "found 2 classes / 140 images")
    a = argparse.Namespace(data=data, arch="resnet18", no_pretrained=True, size=64, batch=16, lr=2e-3, wd=1e-4, head_epochs=1, epochs=20, patience=8, val=0.25, seed=1, workers=0, device=args.device, name="selftest")
    run, best = train(a)
    C.check(best["accuracy"] > 0.75, "validation accuracy %.3f > 0.75 (random init, 64px)" % best["accuracy"])
    res = predict(argparse.Namespace(image=items[-1][0], run=run, out=os.path.join(tmp, "cam.png"), device=args.device))
    C.check(os.path.exists(res["gradcam"]) and res["label"] in classes, "prediction + Grad-CAM overlay written")
    C.log("ALL IMAGE CLASSIFICATION SELF-TESTS PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--data", required=True); t.add_argument("--arch", choices=["resnet18", "resnet50"], default="resnet50")
    t.add_argument("--no-pretrained", dest="no_pretrained", action="store_true"); t.add_argument("--size", type=int, default=224); t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-4); t.add_argument("--wd", type=float, default=1e-4); t.add_argument("--head-epochs", dest="head_epochs", type=int, default=2)
    t.add_argument("--epochs", type=int, default=15); t.add_argument("--patience", type=int, default=4); t.add_argument("--val", type=float, default=0.2)
    t.add_argument("--seed", type=int, default=42); t.add_argument("--workers", type=int, default=0); t.add_argument("--name"); t.add_argument("--device")
    p = sub.add_parser("predict"); p.add_argument("--image", required=True); p.add_argument("--run", default="latest"); p.add_argument("--out"); p.add_argument("--device")
    s = sub.add_parser("selftest"); s.add_argument("--device")
    a = ap.parse_args()
    {"train": train, "predict": predict, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
