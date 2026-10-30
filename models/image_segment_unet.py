"""Image segmentation with U-Net (segmentation-models-pytorch, ResNet-34 encoder with ImageNet weights).

  python models/image_segment_unet.py train   --images path\\to\\images --masks path\\to\\masks
  python models/image_segment_unet.py predict --image scan.png [--run latest]   -> mask PNG + overlay PNG
  python models/image_segment_unet.py selftest   (offline: generated images/masks, no weight download)

Masks: same file name (any image extension) as the image; pixel value 0 = background, 1..K = classes
(a 0/255 binary mask is read as 0/1). Loss = Dice + cross-entropy; metrics = mean IoU and Dice per class.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

TASK = "segment"
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def pairs(img_dir, mask_dir):
    masks = {os.path.splitext(f)[0]: os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.lower().endswith(EXTS)}
    out = [(os.path.join(img_dir, f), masks[os.path.splitext(f)[0]]) for f in sorted(os.listdir(img_dir)) if f.lower().endswith(EXTS) and os.path.splitext(f)[0] in masks]
    if not out:
        sys.exit("No image/mask pairs with matching file names found.")
    return out


def read_mask(p, size):
    from PIL import Image
    with Image.open(p) as m:
        a = np.asarray(m.convert("L").resize((size, size), Image.NEAREST)).astype(np.int64)
    if a.max() == 255 and len(np.unique(a)) <= 2:
        a = (a > 127).astype(np.int64)
    return a


def read_image(p, size):
    from PIL import Image
    with Image.open(p) as im:
        a = np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR)).astype(np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1)


class PairSet:
    def __init__(self, items, size, n_classes, train):
        self.items, self.size, self.k, self.train = items, size, n_classes, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch
        x, m = read_image(self.items[i][0], self.size), np.clip(read_mask(self.items[i][1], self.size), 0, self.k - 1)
        if self.train:
            if np.random.rand() < 0.5:
                x, m = x[:, :, ::-1], m[:, ::-1]
            if np.random.rand() < 0.5:
                x, m = x[:, ::-1, :], m[::-1, :]
            k = np.random.randint(4)
            x, m = np.rot90(x, k, (1, 2)), np.rot90(m, k)
        return torch.as_tensor(np.ascontiguousarray(x)), torch.as_tensor(np.ascontiguousarray(m))


def make_model(n_classes, encoder, pretrained):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=n_classes)


def iou_dice(pred, true, k):
    ious, dices = [], []
    for c in range(1, k) if k > 1 else [0]:
        p, t = pred == c, true == c
        inter, union, s = np.logical_and(p, t).sum(), np.logical_or(p, t).sum(), p.sum() + t.sum()
        if union == 0:
            continue
        ious.append(inter / union); dices.append(2 * inter / s)
    return float(np.mean(ious)) if ious else 1.0, float(np.mean(dices)) if dices else 1.0


def evaluate(model, loader, device, k):
    import torch
    model.eval(); I, D = [], []
    with torch.no_grad():
        for x, m in loader:
            pred = model(x.to(device)).argmax(1).cpu().numpy()
            for a, b in zip(pred, m.numpy()):
                i, d = iou_dice(a, b, k); I.append(i); D.append(d)
    return float(np.mean(I)), float(np.mean(D))


def train(args):
    import torch
    import segmentation_models_pytorch as smp
    from torch.utils.data import DataLoader
    C.seed_all(args.seed)
    items = pairs(args.images, args.masks)
    k = args.classes or int(max(read_mask(m, 64).max() for _, m in items[:200])) + 1
    k = max(k, 2)
    idx = np.random.default_rng(args.seed).permutation(len(items)); nv = max(1, int(len(items) * args.val))
    va, tr = [items[i] for i in idx[:nv]], [items[i] for i in idx[nv:]]
    device, dname = C.get_device(args.device)
    C.log("%d pairs (%d train / %d val) · %d classes incl. background · %s · device %s" % (len(items), len(tr), len(va), k, args.encoder, dname))
    dl_tr = DataLoader(PairSet(tr, args.size, k, True), batch_size=args.batch, shuffle=True, num_workers=args.workers)
    dl_va = DataLoader(PairSet(va, args.size, k, False), batch_size=args.batch, num_workers=args.workers)
    model = make_model(k, args.encoder, not args.no_pretrained).to(device)
    dice = smp.losses.DiceLoss("multiclass"); ce = torch.nn.CrossEntropyLoss()
    opt = C.make_optimizer(model.parameters(), args.lr, args.wd, dname)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, args.epochs))
    out = C.new_run_dir(TASK, args.name)
    stopper, best, hist = C.EarlyStopper(args.patience, "max"), None, []
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for x, m in dl_tr:
            x, m = x.to(device), m.to(device); o = model(x)
            loss = dice(o, m) + ce(o, m)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(x)
        sched.step()
        miou, mdice = evaluate(model, dl_va, device, k)
        hist.append({"epoch": ep, "loss": tot / len(tr), "val_iou": miou, "val_dice": mdice})
        C.log("epoch %d  loss %.4f  val IoU %.3f  Dice %.3f" % (ep, tot / len(tr), miou, mdice))
        if stopper.step(mdice):
            best = {"iou": miou, "dice": mdice, "epoch": ep}
            torch.save({"state_dict": C.cpu_state(model), "classes": k, "encoder": args.encoder, "size": args.size}, os.path.join(out, "model.pt"))
        if stopper.stop:
            C.log("early stop"); break
    C.save_json(os.path.join(out, "report.json"), {"validation": best, "history": hist, "classes": k, "device": dname})
    C.log("best val Dice %.3f IoU %.3f · saved to %s" % (best["dice"], best["iou"], out))
    return out, best


def predict(args):
    import torch
    from PIL import Image
    run = C.resolve_run(TASK, args.run)
    device, dname = C.get_device(args.device)
    ck = C.load_ckpt(os.path.join(run, "model.pt"))
    model = make_model(ck["classes"], ck["encoder"], False); model.load_state_dict(ck["state_dict"]); model.to(device).eval()
    x = torch.as_tensor(read_image(args.image, ck["size"]))[None].to(device)
    with torch.no_grad():
        prob = torch.softmax(model(x), 1)[0].cpu().numpy()
    mask = prob.argmax(0).astype(np.uint8)
    stem = args.out or os.path.splitext(args.image)[0]
    Image.fromarray((mask * (255 // max(1, ck["classes"] - 1))).astype(np.uint8)).save(stem + "_mask.png")
    with Image.open(args.image) as im:
        base = np.asarray(im.convert("RGB").resize((ck["size"], ck["size"]))).astype(np.float32)
    colors = np.array([[0, 0, 0], [255, 80, 120], [60, 200, 180], [242, 180, 90], [90, 180, 255]], np.float32)
    over = np.where(mask[..., None] > 0, base * 0.5 + colors[np.minimum(mask, 4)] * 0.5, base)
    Image.fromarray(over.clip(0, 255).astype(np.uint8)).save(stem + "_overlay.png")
    area = {int(c): float((mask == c).mean()) for c in range(1, ck["classes"])}
    C.log("mask -> %s_mask.png · overlay -> %s_overlay.png · area fraction per class %s" % (stem, stem, {c: round(v, 4) for c, v in area.items()}))
    return {"mask": stem + "_mask.png", "overlay": stem + "_overlay.png", "area": area}


def synthetic(root, n=60, size=96, seed=0):
    from PIL import Image
    rng = np.random.default_rng(seed); os.makedirs(os.path.join(root, "images"), exist_ok=True); os.makedirs(os.path.join(root, "masks"), exist_ok=True)
    yy, xx = np.mgrid[0:size, 0:size]
    for i in range(n):
        v = 0.3 + 0.15 * np.sin(yy / 6 + rng.random() * 6) + rng.normal(0, .06, (size, size)); m = np.zeros((size, size), np.uint8)
        for _ in range(rng.integers(1, 3)):
            cx, cy = rng.integers(20, size - 20, 2); ax, ay = rng.integers(8, 18, 2)
            blob = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 <= 1
            v = np.where(blob, v + 0.35, v); m[blob] = 255
        Image.fromarray((np.clip(v, 0, 1) * 255).astype(np.uint8), "L").save(os.path.join(root, "images", "%03d.png" % i))
        Image.fromarray(m, "L").save(os.path.join(root, "masks", "%03d.png" % i))


def selftest(args):
    import tempfile
    tmp = tempfile.mkdtemp(prefix="nf_seg_"); synthetic(tmp)
    items = pairs(os.path.join(tmp, "images"), os.path.join(tmp, "masks"))
    C.check(len(items) == 60 and set(np.unique(read_mask(items[0][1], 96))) <= {0, 1}, "60 image/mask pairs, binary masks read as 0/1")
    a = argparse.Namespace(images=os.path.join(tmp, "images"), masks=os.path.join(tmp, "masks"), classes=None, encoder="resnet18", no_pretrained=True, size=96, batch=8,
                           lr=1e-3, wd=1e-4, epochs=15, patience=6, val=0.2, seed=2, workers=0, device=args.device, name="selftest")
    run, best = train(a)
    C.check(best["dice"] > 0.6, "validation Dice %.3f > 0.60 (random init)" % best["dice"])
    res = predict(argparse.Namespace(image=items[0][0], run=run, out=os.path.join(tmp, "pred"), device=args.device))
    C.check(os.path.exists(res["overlay"]), "mask and overlay written")
    C.log("ALL SEGMENTATION SELF-TESTS PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--images", required=True); t.add_argument("--masks", required=True); t.add_argument("--classes", type=int)
    t.add_argument("--encoder", default="resnet34"); t.add_argument("--no-pretrained", dest="no_pretrained", action="store_true"); t.add_argument("--size", type=int, default=256)
    t.add_argument("--batch", type=int, default=8); t.add_argument("--lr", type=float, default=3e-4); t.add_argument("--wd", type=float, default=1e-4)
    t.add_argument("--epochs", type=int, default=40); t.add_argument("--patience", type=int, default=8); t.add_argument("--val", type=float, default=0.2)
    t.add_argument("--seed", type=int, default=42); t.add_argument("--workers", type=int, default=0); t.add_argument("--name"); t.add_argument("--device")
    p = sub.add_parser("predict"); p.add_argument("--image", required=True); p.add_argument("--run", default="latest"); p.add_argument("--out"); p.add_argument("--device")
    s = sub.add_parser("selftest"); s.add_argument("--device")
    a = ap.parse_args()
    {"train": train, "predict": predict, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
