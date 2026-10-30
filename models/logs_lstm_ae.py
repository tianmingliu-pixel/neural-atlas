"""Log / traffic anomaly detection with an LSTM autoencoder (PyTorch).

  python models/logs_lstm_ae.py train   --baseline normal_access.log            (or a numeric CSV)
  python models/logs_lstm_ae.py score   --input to_check.log [--labels labels.txt] [--run latest]
  python models/logs_lstm_ae.py selftest

Input: Apache/Nginx access logs (common/combined format) or a numeric CSV (e.g. NetFlow exports).
Each line becomes a feature vector (same 20 features as the Neural Forge web page for access logs).
Consecutive lines are grouped into sliding windows; the autoencoder learns to reconstruct windows of
normal traffic. A line's anomaly score is its reconstruction error averaged over the windows that
contain it. Threshold = a percentile of the baseline scores (default 99.5).
`--labels` (optional): one 0/1 per line of --input, to report AUC / precision / recall.
"""
import argparse
import math
import os
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

TASK = "logs"
LOG_RE = re.compile(r'^(\S+) \S+ \S+ \[([^\]]+)\] "(\S+) (\S+)(?: [^"]*)?" (\d{3}) (\d+|-)(?: "([^"]*)" "([^"]*)")?')
FEATS = ["method_get", "method_post", "method_other", "status_2xx", "status_3xx", "status_4xx", "status_5xx", "log_bytes", "path_len", "query_len",
         "path_depth", "n_params", "pct_encoded", "special_ratio", "path_entropy", "ua_len", "ua_nonbrowser", "hour_sin", "hour_cos", "ip_rate"]


def entropy(s):
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def line_features(line):
    m = LOG_RE.match(line.strip())
    if not m:
        return None
    ip, tm, method, url, st, nbytes, ua = m.group(1), m.group(2), m.group(3), m.group(4), int(m.group(5)), m.group(6), m.group(8) or ""
    path, _, query = url.partition("?")
    hm = re.search(r":(\d{2}):\d{2}:\d{2}", tm); hour = int(hm.group(1)) if hm else 12
    special = len(re.findall(r"[^A-Za-z0-9/._\-?=&+]", url))
    f = [method == "GET", method == "POST", method not in ("GET", "POST"), st < 300, 300 <= st < 400, 400 <= st < 500, st >= 500,
         math.log1p(0 if nbytes == "-" else int(nbytes)), len(path), len(query), len([p for p in path.split("/") if p]), len(query.split("&")) if query else 0,
         len(re.findall(r"%[0-9a-fA-F]{2}", url)), special / max(1, len(url)), entropy(path), len(ua), 0 if re.search("mozilla", ua, re.I) else 1,
         math.sin(hour / 24 * 2 * math.pi), math.cos(hour / 24 * 2 * math.pi), 0.0]
    return ip, [float(v) for v in f]


def parse(text):
    """Return (kind, feature_names, X[n,d], raw_lines)."""
    lines = [l for l in text.splitlines() if l.strip()]
    parsed = [(l, line_features(l)) for l in lines]
    ok = [(l, r) for l, r in parsed if r]
    if ok and len(ok) >= 0.6 * len(lines):
        cnt = Counter(r[0] for _, r in ok)
        X = np.array([r[1] for _, r in ok], dtype=np.float32)
        X[:, 19] = np.log1p(np.array([cnt[r[0]] for _, r in ok]) * 1000.0 / len(ok))
        return "access", FEATS, X, [l for l, _ in ok]
    import csv, io
    rows = list(csv.reader(io.StringIO(text.strip())))
    if len(rows) < 3:
        sys.exit("Could not parse the input as access logs or as a CSV with a header.")
    head, body = rows[0], rows[1:]
    def num(v):
        try:
            return float(v)
        except ValueError:
            return None
    cols = [j for j in range(len(head)) if sum(num(r[j]) is not None for r in body if j < len(r)) >= 0.9 * len(body)]
    X = np.array([[num(r[j]) if j < len(r) and num(r[j]) is not None else 0.0 for j in cols] for r in body], dtype=np.float32)
    return "csv", [head[j] for j in cols], X, [",".join(r) for r in body]


def windows(X, W):
    if len(X) < W:
        X = np.concatenate([X, np.repeat(X[-1:], W - len(X), 0)])
    return np.stack([X[i:i + W] for i in range(len(X) - W + 1)])


def line_scores(err_w, n, W):
    """err_w: [n_windows, W, d] squared errors -> per-line score [n] and per-line feature errors [n, d]."""
    d = err_w.shape[2]; acc = np.zeros((n, d)); cnt = np.zeros(n)
    for i in range(err_w.shape[0]):
        k = min(W, n - i)
        if k <= 0:
            break
        acc[i:i + k] += err_w[i, :k]; cnt[i:i + k] += 1
    per = acc / np.maximum(cnt, 1)[:, None]
    return per.mean(1), per


# ---------------- model ----------------
def make_model(d, hidden=64, latent=16, layers=1):
    import torch
    import torch.nn as nn

    class LSTMAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.LSTM(d, hidden, num_layers=layers, batch_first=True)
            self.to_z = nn.Linear(hidden, latent)
            self.from_z = nn.Linear(latent, hidden)
            self.dec = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True)
            self.out = nn.Linear(hidden, d)

        def forward(self, x):  # x: [B, W, d]
            _, (h, _) = self.enc(x)
            z = self.to_z(h[-1])                                   # [B, latent]
            rep = self.from_z(z).unsqueeze(1).repeat(1, x.size(1), 1)  # [B, W, hidden]
            y, _ = self.dec(rep)
            return self.out(y)
    return LSTMAE()


def recon_errors(model, Xw, device, batch=256):
    import torch
    model.eval(); outs = []
    with torch.no_grad():
        for s in range(0, len(Xw), batch):
            xb = torch.as_tensor(Xw[s:s + batch], dtype=torch.float32, device=device)
            outs.append(((model(xb) - xb) ** 2).float().cpu().numpy())
    return np.concatenate(outs)


def train(args):
    import torch
    C.seed_all(args.seed)
    with open(args.baseline, encoding="utf-8", errors="replace") as f:
        kind, feats, X, raw = parse(f.read())
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    Xs = np.clip((X - mu) / sd, -8, 8).astype(np.float32)
    W = min(args.window, max(2, len(Xs)))
    Xw = windows(Xs, W)
    rng = np.random.default_rng(args.seed); idx = rng.permutation(len(Xw)); nv = max(1, int(len(idx) * args.val))
    va, tr = idx[:nv], idx[nv:]
    device, dname = C.get_device(args.device)
    C.log("%s input · %d lines · %d features · window %d · %d train / %d val windows · device %s" % (kind, len(X), X.shape[1], W, len(tr), len(va), dname))
    model = make_model(X.shape[1], args.hidden, args.latent, args.layers).to(device)
    opt = C.make_optimizer(model.parameters(), args.lr, args.wd, dname)
    stopper, best, hist = C.EarlyStopper(args.patience, "min"), None, []
    Xt = torch.as_tensor(Xw)
    for ep in range(1, args.epochs + 1):
        model.train(); perm = tr[rng.permutation(len(tr))]; tot = 0.0
        for s in range(0, len(perm), args.batch):
            xb = Xt[perm[s:s + args.batch]].to(device)
            loss = ((model(xb) - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); C.clip_grads(model.parameters(), 1.0, dname); opt.step(); tot += float(loss) * len(xb)
        vloss = float(recon_errors(model, Xw[va], device).mean())
        hist.append({"epoch": ep, "loss": tot / max(1, len(tr)), "val_loss": vloss})
        if stopper.step(vloss):
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep % max(1, args.epochs // 10) == 0:
            C.log("epoch %3d  loss %.4f  val %.4f" % (ep, tot / max(1, len(tr)), vloss))
        if stopper.stop:
            C.log("early stop at epoch %d" % ep); break
    model.load_state_dict(best)
    base_scores, _ = line_scores(recon_errors(model, Xw, device), len(Xs), W)
    thr = float(np.percentile(base_scores, args.percentile))
    out = C.new_run_dir(TASK, args.name)
    torch.save({"state_dict": best, "d": X.shape[1], "hidden": args.hidden, "latent": args.latent, "layers": args.layers}, os.path.join(out, "model.pt"))
    C.save_json(os.path.join(out, "meta.json"), {"kind": kind, "features": feats, "mu": mu.tolist(), "sd": sd.tolist(), "window": W, "threshold": thr,
                                                  "percentile": args.percentile, "baseline_scores": np.sort(base_scores).tolist()})
    C.save_json(os.path.join(out, "report.json"), {"history": hist, "n_lines": len(X), "threshold": thr, "device": dname})
    C.log("threshold (p%.1f of baseline) = %.4f · saved to %s" % (args.percentile, thr, out))
    return out


def score(args):
    import torch
    run = C.resolve_run(TASK, args.run)
    meta = C.load_json(os.path.join(run, "meta.json"))
    device, dname = C.get_device(args.device)
    ck = C.load_ckpt(os.path.join(run, "model.pt"))
    model = make_model(ck["d"], ck["hidden"], ck["latent"], ck["layers"]); model.load_state_dict(ck["state_dict"]); model.to(device)
    with open(args.input, encoding="utf-8", errors="replace") as f:
        kind, feats, X, raw = parse(f.read())
    if kind != meta["kind"] or feats != meta["features"]:
        sys.exit("Input format (%s, %d features) does not match the trained model (%s, %d features)." % (kind, len(feats), meta["kind"], len(meta["features"])))
    if kind == "access" and len(X) < 50:
        X[:, 19] = meta["mu"][19]  # per-IP rate is unreliable on tiny inputs: use the baseline mean
    mu, sd = np.array(meta["mu"], np.float32), np.array(meta["sd"], np.float32)
    Xs = np.clip((X - mu) / sd, -8, 8).astype(np.float32)
    W = meta["window"]
    s, per = line_scores(recon_errors(model, windows(Xs, W), device), len(Xs), W)
    thr = meta["threshold"] if args.percentile is None else float(np.percentile(meta["baseline_scores"], args.percentile))
    order = np.argsort(-s)
    rows = [{"rank": int(r + 1), "line_no": int(i + 1), "score": round(float(s[i]), 5), "anomalous": bool(s[i] > thr),
             "top_features": [feats[j] for j in np.argsort(-per[i])[:3]], "line": raw[i][:400]} for r, i in enumerate(order)]
    res = {"threshold": thr, "n": len(s), "flagged": int((s > thr).sum()), "top": rows[:args.top]}
    if args.labels:
        with open(args.labels, encoding="utf-8") as f:
            lab = np.array([int(x.strip()) for x in f if x.strip()][:len(s)])
        from sklearn.metrics import precision_score, recall_score, roc_auc_score
        pred = (s[:len(lab)] > thr).astype(int)
        res["auc"] = float(roc_auc_score(lab, s[:len(lab)])) if 0 < lab.sum() < len(lab) else None
        res["precision"] = float(precision_score(lab, pred, zero_division=0)); res["recall"] = float(recall_score(lab, pred, zero_division=0))
        C.log("AUC %s · precision %.3f · recall %.3f" % (("%.3f" % res["auc"]) if res["auc"] is not None else "n/a", res["precision"], res["recall"]))
    out = args.out or os.path.join(run, "scores.json"); C.save_json(out, res)
    C.log("%d of %d lines above threshold %.4f · wrote %s" % (res["flagged"], res["n"], thr, out))
    for r in rows[:min(10, args.top)]:
        print("  #%-3d score %.4f %s %s\n        %s" % (r["rank"], r["score"], "ANOMALY" if r["anomalous"] else "normal ", r["top_features"], r["line"][:160]))
    return res


# ---------------- self-test (benign synthetic traffic; anomalies are statistical, not attack strings) ----------------
def synthetic(seed=21):
    rng = np.random.default_rng(seed)
    ips = ["10.%d.%d.%d" % tuple(rng.integers(0, 255, 3)) for _ in range(80)]
    uas = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/129.0", "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) Safari/605.1.15", "Mozilla/5.0 (X11; Linux x86_64; rv:131.0) Firefox/131.0"]
    def ts(h):
        return "[%02d/Oct/2026:%02d:%02d:%02d -0700]" % (rng.integers(1, 28), h, rng.integers(0, 60), rng.integers(0, 60))
    def line(ip, h, m, u, st, b, ua):
        return '%s - - %s "%s %s HTTP/1.1" %d %d "-" "%s"' % (ip, ts(h), m, u, st, b, ua)
    def normal():
        ip, ua, h, r = ips[rng.integers(len(ips))], uas[rng.integers(len(uas))], int(rng.integers(8, 22)) if rng.random() < .85 else int(rng.integers(0, 24)), rng.random()
        if r < .3: return line(ip, h, "GET", "/products/%d" % rng.integers(100, 999), 200, rng.integers(8000, 24000), ua)
        if r < .5: return line(ip, h, "GET", "/static/js/app.3f9c1.js", 200, rng.integers(90000, 180000), ua)
        if r < .7: return line(ip, h, "GET", "/api/v1/items?page=%d&size=20" % rng.integers(1, 30), 200, rng.integers(1500, 9000), ua)
        if r < .85: return line(ip, h, "POST", "/api/v1/cart", 201, rng.integers(200, 900), ua)
        return line(ip, h, "GET", "/", 200, rng.integers(14000, 18000), ua)
    def unusual():
        r, h = rng.random(), int(rng.integers(1, 5))
        rnd = lambda n: "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), n))
        if r < .35: return line("10.99.7.13", h, "GET", "/%s/%s" % (rnd(10), rnd(8)), 404, rng.integers(150, 300), "custom-client/0.9")
        if r < .55: return line(ips[0], int(rng.integers(0, 24)), "GET", "/api/v1/export?range=" + rnd(40), 500, rng.integers(120, 400), uas[0])
        if r < .75: return line(ips[1], h, "GET", "/api/v1/items?page=1&size=100000", 200, rng.integers(40000000, 90000000), "batch-job/2.1")
        return line(ips[2], int(rng.integers(0, 24)), "DELETE", "/products/%d" % rng.integers(100, 999), 405, 200, "custom-client/0.9")
    base = [normal() for _ in range(900)]
    test, lab = [], []
    for i in range(300):
        if i % 25 == 12:
            test.append(unusual()); lab.append(1)
        else:
            test.append(normal()); lab.append(0)
    return base, test, lab


def selftest(args):
    import tempfile
    tmp = tempfile.mkdtemp(prefix="nf_logs_")
    base, test, lab = synthetic()
    kind, feats, X, _ = parse("\n".join(base))
    C.check(kind == "access" and X.shape == (900, 20), "parsed 900 access-log lines into 20 features")
    pb, pt, pl = (os.path.join(tmp, n) for n in ("base.log", "test.log", "labels.txt"))
    open(pb, "w").write("\n".join(base)); open(pt, "w").write("\n".join(test)); open(pl, "w").write("\n".join(map(str, lab)))
    a = argparse.Namespace(baseline=pb, window=16, hidden=64, latent=16, layers=1, epochs=40, batch=64, lr=2e-3, wd=1e-5, val=0.15, patience=8, percentile=99.5, seed=3, device=args.device, name="selftest")
    run = train(a)
    res = score(argparse.Namespace(input=pt, labels=pl, run=run, percentile=None, top=25, out=os.path.join(tmp, "scores.json"), device=args.device))
    C.check(res["auc"] is not None and res["auc"] > 0.9, "AUC %.3f > 0.90 on held-out traffic" % (res["auc"] or 0))
    C.check(res["recall"] > 0.7, "recall %.3f > 0.70 at the baseline p99.5 threshold" % res["recall"])
    C.log("ALL LOG SELF-TESTS PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--baseline", required=True); t.add_argument("--window", type=int, default=16)
    t.add_argument("--hidden", type=int, default=64); t.add_argument("--latent", type=int, default=16); t.add_argument("--layers", type=int, default=1)
    t.add_argument("--epochs", type=int, default=60); t.add_argument("--batch", type=int, default=64); t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--wd", type=float, default=1e-5); t.add_argument("--val", type=float, default=0.15); t.add_argument("--patience", type=int, default=8)
    t.add_argument("--percentile", type=float, default=99.5); t.add_argument("--seed", type=int, default=42); t.add_argument("--name"); t.add_argument("--device")
    s = sub.add_parser("score"); s.add_argument("--input", required=True); s.add_argument("--labels"); s.add_argument("--run", default="latest")
    s.add_argument("--percentile", type=float); s.add_argument("--top", type=int, default=50); s.add_argument("--out"); s.add_argument("--device")
    st = sub.add_parser("selftest"); st.add_argument("--device")
    a = ap.parse_args()
    {"train": train, "score": score, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
