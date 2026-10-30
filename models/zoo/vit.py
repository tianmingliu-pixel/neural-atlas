"""Vision Transformer (Dosovitskiy et al., 2020) with timm.

  python models/zoo/vit.py train    --data path\\to\\images (one sub-folder per class) [--model vit_small_patch16_224]
  python models/zoo/vit.py predict  --image photo.png
  python models/zoo/vit.py selftest            (small ViT from scratch on the bundled digits, offline)

Images are cut into patches, linearly embedded with a [CLS] token and position embeddings, and passed
through Transformer encoder blocks. With --data, ImageNet-pretrained weights are fine-tuned (download on
first use) using the model's own normalization from timm's data config.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE)]
import common as C  # noqa: E402
import _data as D  # noqa: E402


def fit_arrays(model, Xtr, ytr, Xva, yva, args, device, dname):
    import torch
    opt = C.make_optimizer(model.parameters(), args.lr, 0.05, dname)
    steps = args.epochs * ((len(Xtr) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1); Xt, yt = torch.as_tensor(Xtr), torch.as_tensor(ytr)
    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train(); perm = torch.randperm(len(Xt)); tot = 0.0
        for s in range(0, len(Xt), args.batch):
            b = perm[s:s + args.batch]
            loss = loss_fn(model(Xt[b].to(device)), yt[b].to(device))
            opt.zero_grad(); loss.backward(); C.clip_grads(model.parameters(), 1.0, dname); opt.step(); sched.step(); tot += float(loss) * len(b)
        model.eval()
        with torch.no_grad():
            pred = np.concatenate([model(torch.as_tensor(Xva[s:s + 256], device=device)).argmax(1).cpu().numpy() for s in range(0, len(Xva), 256)])
        acc = float((pred == yva).mean()); best = max(best, acc)
        C.log("epoch %d  loss %.4f  val acc %.4f" % (ep, tot / len(Xt), acc))
    return best, pred


def train(args):
    import timm
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    sys.path.insert(0, os.path.dirname(HERE)); import image_classify as IC  # reuse folder listing + dataset
    C.seed_all(args.seed)
    classes, items = IC.list_images(args.data); y = np.array([c for _, c in items]); tr, va = C.stratified_split(y, 0.2, args.seed)
    device, dname = C.get_device(args.device); C.safe_attention(dname)
    model = timm.create_model(args.model, pretrained=not args.no_pretrained, num_classes=len(classes)).to(device)
    from timm.data import resolve_model_data_config; cfg = resolve_model_data_config(model); size = cfg["input_size"][-1]
    norm = T.Normalize(cfg["mean"], cfg["std"])
    tf_tr = T.Compose([T.Lambda(IC.to_rgb), T.RandomResizedCrop(size, scale=(0.7, 1)), T.RandomHorizontalFlip(), T.ToTensor(), norm])
    tf_va = T.Compose([T.Lambda(IC.to_rgb), T.Resize(int(size * 1.14)), T.CenterCrop(size), T.ToTensor(), norm])
    dl_tr = DataLoader(IC.Files([items[i] for i in tr], tf_tr), batch_size=args.batch, shuffle=True)
    dl_va = DataLoader(IC.Files([items[i] for i in va], tf_va), batch_size=args.batch)
    opt = C.make_optimizer(model.parameters(), args.lr, 0.05, dname); loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    out = D.out_dir("vit"); best = -1
    C.log("%s · %d classes · %d train / %d val · input %d · device %s" % (args.model, len(classes), len(tr), len(va), size, dname))
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, yb in dl_tr:
            loss = loss_fn(model(x.to(device)), yb.to(device)); opt.zero_grad(); loss.backward(); opt.step()
        yt_, yp_ = IC.run_eval(model, dl_va, device); rep = C.cls_report(yt_, yp_, classes)
        C.log("epoch %d  val acc %.3f  F1 %.3f" % (ep, rep["accuracy"], rep["macro_f1"]))
        if rep["macro_f1"] > best:
            best = rep["macro_f1"]; torch.save({"state_dict": C.cpu_state(model), "model": args.model, "classes": classes}, os.path.join(out, "model.pt"))
            C.save_json(os.path.join(out, "report.json"), {"validation": rep})
    return best


def predict(args):
    import timm
    import torch
    from PIL import Image
    from torchvision import transforms as T
    device, _ = C.get_device(args.device); ck = C.load_ckpt(os.path.join(D.out_dir("vit"), "model.pt"))
    model = timm.create_model(ck["model"], pretrained=False, num_classes=len(ck["classes"])); model.load_state_dict(ck["state_dict"]); model.to(device).eval()
    from timm.data import resolve_model_data_config; cfg = resolve_model_data_config(model); size = cfg["input_size"][-1]
    tf = T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size), T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])
    with Image.open(args.image) as im:
        x = tf(im.convert("RGB"))[None].to(device)
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0].cpu().numpy()
    C.log("%s -> %s (%.3f)" % (args.image, ck["classes"][int(p.argmax())], p.max())); return p


def selftest(args):
    import timm
    C.seed_all(0)
    X, y = D.digits(32, scale="minus1_1"); tr, va = C.stratified_split(y, 0.2, 0)
    device, dname = C.get_device(args.device); C.safe_attention(dname)
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10, img_size=32, patch_size=4, in_chans=1, depth=4); model.to(device)
    C.log("ViT-tiny (4 blocks, 4x4 patches) · %d params · device %s" % (sum(p.numel() for p in model.parameters()), dname))
    best, _ = fit_arrays(model, X[tr], y[tr], X[va], y[va], argparse.Namespace(lr=2e-3, epochs=40, batch=64), device, dname)
    C.check(best > 0.8, "ViT from scratch, digits val accuracy %.3f > 0.80" % best)
    C.log("VIT SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--data", required=True); t.add_argument("--model", default="vit_small_patch16_224"); t.add_argument("--no-pretrained", dest="no_pretrained", action="store_true")
    t.add_argument("--epochs", type=int, default=10); t.add_argument("--batch", type=int, default=16); t.add_argument("--lr", type=float, default=5e-5); t.add_argument("--seed", type=int, default=0); t.add_argument("--device")
    p = sub.add_parser("predict"); p.add_argument("--image", required=True); p.add_argument("--device")
    s = sub.add_parser("selftest"); s.add_argument("--device")
    a = ap.parse_args(); {"train": train, "predict": predict, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
