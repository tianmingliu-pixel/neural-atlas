"""Recurrent neural network (Elman RNN, torch.nn.RNN) for time-series forecasting.

  python models/zoo/rnn.py train    [--csv series.csv --column value] [--window 48] [--horizon 24]
  python models/zoo/rnn.py selftest            (synthetic seasonal series with trend and noise)

The RNN reads the last `window` values and predicts the next one. It is compared with the naive
"repeat the last value" baseline; a forecast of `horizon` future steps is written to forecast.csv.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE)]
import common as C  # noqa: E402
import _data as D  # noqa: E402


def synthetic(n=1500, seed=0):
    rng = np.random.default_rng(seed); t = np.arange(n)
    return (10 + 0.01 * t + 3 * np.sin(2 * np.pi * t / 24) + 1.5 * np.sin(2 * np.pi * t / 168) + rng.normal(0, 0.4, n)).astype(np.float32)


def make_model(hidden, layers, cell="rnn"):
    import torch.nn as nn

    class Forecaster(nn.Module):
        def __init__(self):
            super().__init__()
            rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[cell]
            self.rnn = rnn_cls(1, hidden, num_layers=layers, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):          # x: [B, W, 1]
            y, _ = self.rnn(x)
            return self.head(y[:, -1])  # next value
    return Forecaster()


def windows(s, W):
    X = np.stack([s[i:i + W] for i in range(len(s) - W)])[..., None]
    return X.astype(np.float32), s[W:].astype(np.float32)[:, None]


def train(args):
    import torch
    C.seed_all(args.seed)
    if args.csv:
        with open(args.csv, encoding="utf-8") as f:
            series = np.array([float(r[args.column]) for r in csv.DictReader(f) if r.get(args.column) not in (None, "")], np.float32)
    else:
        series = synthetic()
    cut = int(len(series) * 0.8); mu, sd = series[:cut].mean(), series[:cut].std() or 1.0
    z = (series - mu) / sd
    Xtr, ytr = windows(z[:cut], args.window); Xte, yte = windows(z[cut - args.window:], args.window)
    device, dname = C.get_device(args.device)
    model = make_model(args.hidden, args.layers, args.cell).to(device)
    opt = C.make_optimizer(model.parameters(), args.lr, 1e-5, dname); loss_fn = torch.nn.MSELoss()
    Xt, yt = torch.as_tensor(Xtr), torch.as_tensor(ytr)
    C.log("%s · %d train windows / %d test · window %d · device %s" % (args.cell.upper(), len(Xtr), len(Xte), args.window, dname))
    def mae_model():
        model.eval()
        with torch.no_grad():
            p = model(torch.as_tensor(Xte, device=device)).cpu().numpy()
        return float(np.abs(p - yte).mean() * sd)
    for ep in range(1, args.epochs + 1):
        model.train(); perm = torch.randperm(len(Xt)); tot = 0.0
        for s in range(0, len(Xt), args.batch):
            b = perm[s:s + args.batch]
            loss = loss_fn(model(Xt[b].to(device)), yt[b].to(device))
            opt.zero_grad(); loss.backward(); C.clip_grads(model.parameters(), 1.0, dname); opt.step(); tot += float(loss) * len(b)
        if ep == 1 or ep % max(1, args.epochs // 8) == 0:
            C.log("epoch %d  loss %.4f  test MAE %.4f" % (ep, tot / len(Xt), mae_model()))
    mae = mae_model(); naive = float(np.abs(Xte[:, -1, 0] - yte[:, 0]).mean() * sd)
    C.log("test MAE: %s %.4f vs naive last-value %.4f" % (args.cell.upper(), mae, naive))
    # recursive multi-step forecast from the end of the series
    hist = list(z[-args.window:]); fc = []
    model.eval()
    with torch.no_grad():
        for _ in range(args.horizon):
            v = float(model(torch.as_tensor(np.array(hist[-args.window:], np.float32)[None, :, None], device=device)).cpu()[0, 0])
            fc.append(v); hist.append(v)
    out = D.out_dir("rnn")
    with open(os.path.join(out, "forecast.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "forecast"]); w.writerows([[i + 1, round(v * sd + mu, 4)] for i, v in enumerate(fc)])
    torch.save({"state_dict": C.cpu_state(model), "hidden": args.hidden, "layers": args.layers, "cell": args.cell, "window": args.window, "mu": float(mu), "sd": float(sd)}, os.path.join(out, "model.pt"))
    C.save_json(os.path.join(out, "report.json"), {"test_mae": mae, "naive_mae": naive, "cell": args.cell})
    C.log("forecast of %d steps -> %s" % (args.horizon, os.path.join(out, "forecast.csv")))
    return mae, naive


def selftest(args):
    mae, naive = train(argparse.Namespace(csv=None, column=None, window=48, horizon=24, hidden=64, layers=2, cell="rnn", epochs=25, batch=64, lr=3e-3, seed=0, device=args.device))
    C.check(mae < naive, "RNN test MAE %.3f beats the naive baseline %.3f" % (mae, naive))
    C.log("RNN SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--csv"); t.add_argument("--column", default="value"); t.add_argument("--window", type=int, default=48)
    t.add_argument("--horizon", type=int, default=24); t.add_argument("--hidden", type=int, default=64); t.add_argument("--layers", type=int, default=2)
    t.add_argument("--cell", choices=["rnn", "gru", "lstm"], default="rnn"); t.add_argument("--epochs", type=int, default=40); t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-3); t.add_argument("--seed", type=int, default=0); t.add_argument("--device")
    s = sub.add_parser("selftest"); s.add_argument("--device")
    a = ap.parse_args(); {"train": train, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
