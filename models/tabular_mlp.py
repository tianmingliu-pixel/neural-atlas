"""Tabular prediction with an MLP (PyTorch) + scikit-learn preprocessing.

  python models/tabular_mlp.py train   --csv data.csv --target failure_within_7d
  python models/tabular_mlp.py predict --csv new.csv [--run latest] [--out predictions.csv]
  python models/tabular_mlp.py selftest

Numeric columns: median imputation + standardization. Categorical columns: most-frequent imputation +
one-hot (unknown categories ignored). Classification when the target is non-numeric or has <= 10 integer
values; otherwise regression. Early stopping on the validation metric. Permutation importance per column.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

TASK = "tabular"


# ---------------- data ----------------
def infer_task(y):
    if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y):
        return "cls"
    v = y.dropna()
    if len(v) and np.all(np.equal(np.mod(v, 1), 0)) and v.nunique() <= 10:
        return "cls"
    return "reg"


def build_preprocessor(df, target, max_categories=40):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    feats = [c for c in df.columns if c != target]
    num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in feats if c not in num and df[c].nunique(dropna=True) <= max_categories]
    dropped = [c for c in feats if c not in num and c not in cat]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ], sparse_threshold=0.0)
    return pre, num, cat, dropped


# ---------------- model ----------------
def make_model(n_in, n_out, hidden=(64, 32), dropout=0.15):
    import torch.nn as nn
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


def predict_raw(model, X, device, batch=1024):
    import torch
    model.eval(); outs = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            outs.append(model(torch.as_tensor(X[s:s + batch], dtype=torch.float32, device=device)).float().cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, 1), np.float32)


def metric(task, y_true, raw, tstats=None):
    if task == "cls":
        return float((raw.argmax(1) == y_true).mean())
    from sklearn.metrics import r2_score
    return float(r2_score(y_true, raw[:, 0] * tstats[1] + tstats[0]))


def train(args):
    import joblib
    import torch
    import torch.nn as nn
    C.seed_all(args.seed)
    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        sys.exit("Target column '%s' not found. Columns: %s" % (args.target, list(df.columns)))
    df = df[df[args.target].notna()].reset_index(drop=True)
    task = args.task or infer_task(df[args.target])
    pre, num, cat, dropped = build_preprocessor(df, args.target)
    if dropped:
        C.log("dropped high-cardinality text columns:", dropped)
    if task == "cls":
        classes = sorted(df[args.target].astype(str).unique().tolist())
        y = df[args.target].astype(str).map({c: i for i, c in enumerate(classes)}).values
        tr, va = C.stratified_split(y, args.val, args.seed)
        tstats = None
    else:
        classes = None
        y = df[args.target].astype(float).values
        tr, va = C.stratified_split(np.zeros(len(y)), args.val, args.seed)
        tstats = (float(y[tr].mean()), float(y[tr].std() or 1.0))
    Xtr = pre.fit_transform(df.iloc[tr]).astype(np.float32)
    Xva = pre.transform(df.iloc[va]).astype(np.float32)
    device, dname = C.get_device(args.device)
    C.log("task %s · %d train / %d val · %d features · device %s" % (task, len(tr), len(va), Xtr.shape[1], dname))
    n_out = len(classes) if task == "cls" else 1
    model = make_model(Xtr.shape[1], n_out, tuple(args.hidden), args.dropout).to(device)
    opt = C.make_optimizer(model.parameters(), args.lr, args.wd, dname)
    if task == "cls":
        w = torch.as_tensor(C.class_weights(y[tr], n_out), device=device)
        loss_fn = nn.CrossEntropyLoss(weight=w)
        ytr = torch.as_tensor(y[tr], dtype=torch.long)
    else:
        loss_fn = nn.MSELoss()
        ytr = torch.as_tensor(((y[tr] - tstats[0]) / tstats[1])[:, None], dtype=torch.float32)
    Xt = torch.as_tensor(Xtr)
    stopper, best_state, hist = C.EarlyStopper(args.patience, "max"), None, []
    for ep in range(1, args.epochs + 1):
        model.train(); perm = torch.randperm(len(Xt)); tot = 0.0
        for s in range(0, len(Xt), args.batch):
            b = perm[s:s + args.batch]
            out = model(Xt[b].to(device)); loss = loss_fn(out, ytr[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(b)
        val = metric(task, y[va], predict_raw(model, Xva, device), tstats)
        hist.append({"epoch": ep, "loss": tot / len(Xt), "val": val})
        if stopper.step(val):
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % max(1, args.epochs // 10) == 0 or ep == 1:
            C.log("epoch %3d  loss %.4f  val %s %.4f" % (ep, tot / len(Xt), "acc" if task == "cls" else "R2", val))
        if stopper.stop:
            C.log("early stop at epoch %d (best %.4f)" % (ep, stopper.best)); break
    model.load_state_dict(best_state)
    raw = predict_raw(model, Xva, device)
    report = {"task": task, "target": args.target, "n_train": int(len(tr)), "n_val": int(len(va)), "history": hist, "device": dname}
    if task == "cls":
        report["validation"] = C.cls_report(y[va], raw.argmax(1), classes); C.print_cls_report(report["validation"])
    else:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        pred = raw[:, 0] * tstats[1] + tstats[0]
        report["validation"] = {"r2": float(r2_score(y[va], pred)), "mae": float(mean_absolute_error(y[va], pred)), "rmse": float(np.sqrt(mean_squared_error(y[va], pred)))}
        C.log("R2 %.3f · MAE %.4f · RMSE %.4f" % tuple(report["validation"][k] for k in ("r2", "mae", "rmse")))
    report["importance"] = permutation_importance(model, pre, df.iloc[va], y[va], task, device, tstats, num + cat, args.seed)
    C.log("top features:", ", ".join("%s (%.3f)" % (d["column"], d["drop"]) for d in report["importance"][:5]))
    out = C.new_run_dir(TASK, args.name)
    torch.save({"state_dict": best_state, "n_in": Xtr.shape[1], "n_out": n_out, "hidden": list(args.hidden), "dropout": args.dropout}, os.path.join(out, "model.pt"))
    joblib.dump(pre, os.path.join(out, "preprocessor.joblib"))
    C.save_json(os.path.join(out, "meta.json"), {"task": task, "target": args.target, "classes": classes, "tstats": tstats, "columns": num + cat, "dropped": dropped})
    C.save_json(os.path.join(out, "report.json"), report)
    C.log("saved to", out)
    return out, report


def permutation_importance(model, pre, dfv, yv, task, device, tstats, cols, seed):
    rng = np.random.default_rng(seed)
    base = metric(task, yv, predict_raw(model, pre.transform(dfv).astype(np.float32), device), tstats)
    res = []
    for c in cols:
        d = dfv.copy(); d[c] = rng.permutation(d[c].values)
        res.append({"column": c, "drop": base - metric(task, yv, predict_raw(model, pre.transform(d).astype(np.float32), device), tstats)})
    return sorted(res, key=lambda r: -r["drop"])


def load_run(run, device):
    import joblib
    import torch
    meta = C.load_json(os.path.join(run, "meta.json"))
    ck = C.load_ckpt(os.path.join(run, "model.pt"))
    model = make_model(ck["n_in"], ck["n_out"], tuple(ck["hidden"]), ck["dropout"])
    model.load_state_dict(ck["state_dict"]); model.to(device).eval()
    return model, joblib.load(os.path.join(run, "preprocessor.joblib")), meta


def predict(args):
    run = C.resolve_run(TASK, args.run)
    device, dname = C.get_device(args.device)
    model, pre, meta = load_run(run, device)
    df = pd.read_csv(args.csv)
    raw = predict_raw(model, pre.transform(df).astype(np.float32), device)
    out = df.copy()
    if meta["task"] == "cls":
        z = raw - raw.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        out["prediction"] = [meta["classes"][i] for i in p.argmax(1)]
        for i, c in enumerate(meta["classes"]):
            out["p_" + str(c)] = p[:, i].round(4)
    else:
        out["prediction"] = raw[:, 0] * meta["tstats"][1] + meta["tstats"][0]
    path = args.out or os.path.join(run, "predictions.csv")
    out.to_csv(path, index=False); C.log("wrote", path)
    print(out.head(10).to_string(index=False))
    return out


# ---------------- self-test ----------------
def synthetic(n=600, seed=1):
    rng = np.random.default_rng(seed)
    temp = 55 + rng.normal(0, 8, n); vib = np.clip(2.5 + rng.normal(0, 1.2, n), .2, None); hrs = rng.integers(10, 2200, n)
    load = np.clip(65 + rng.normal(0, 15, n), 20, 100); line = rng.choice(["A", "B", "C"], n); age = rng.integers(1, 16, n)
    z = .12 * (temp - 55) + .9 * (vib - 2.5) + .0012 * (hrs - 1000) + .03 * (load - 65) + (line == "C") * .8 + .08 * (age - 7) - 1.1 + rng.normal(0, .6, n)
    df = pd.DataFrame({"temperature_c": temp.round(1), "vibration_mm_s": vib.round(2), "hours_since_service": hrs, "load_pct": load.round(0), "line": line, "age_years": age,
                       "failure_within_7d": np.where(z > 0, "yes", "no")})
    df.loc[rng.choice(n, 15, replace=False), "load_pct"] = np.nan  # missing values on purpose
    return df


def selftest(args):
    import tempfile
    tmp = tempfile.mkdtemp(prefix="nf_tab_")
    df = synthetic(); p = os.path.join(tmp, "train.csv"); df.to_csv(p, index=False)
    a = argparse.Namespace(csv=p, target="failure_within_7d", task=None, val=0.2, seed=7, epochs=80, batch=32, lr=3e-3, wd=1e-4, hidden=[64, 32], dropout=0.15, patience=12, device=args.device, name="selftest")
    run, rep = train(a)
    C.check(rep["validation"]["accuracy"] > 0.8, "classification accuracy %.3f > 0.80" % rep["validation"]["accuracy"])
    C.check(rep["importance"][0]["column"] in ("vibration_mm_s", "temperature_c", "hours_since_service"), "most important feature is a real driver (%s)" % rep["importance"][0]["column"])
    new = df.drop(columns=["failure_within_7d"]).head(20); q = os.path.join(tmp, "new.csv"); new.to_csv(q, index=False)
    out = predict(argparse.Namespace(csv=q, run=run, out=os.path.join(tmp, "pred.csv"), device=args.device))
    C.check(len(out) == 20 and set(out["prediction"]) <= {"yes", "no"}, "prediction file has 20 labeled rows")
    reg = synthetic(400, 3); reg["risk"] = (reg["temperature_c"] * .5 + reg["vibration_mm_s"] * 3).round(3); reg = reg.drop(columns=["failure_within_7d"])
    p2 = os.path.join(tmp, "reg.csv"); reg.to_csv(p2, index=False)
    a2 = argparse.Namespace(**dict(vars(a), csv=p2, target="risk", name="selftest-reg"))
    _, rep2 = train(a2)
    C.check(rep2["validation"]["r2"] > 0.8, "regression R2 %.3f > 0.80" % rep2["validation"]["r2"])
    C.log("ALL TABULAR SELF-TESTS PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--csv", required=True); t.add_argument("--target", required=True)
    t.add_argument("--task", choices=["cls", "reg"]); t.add_argument("--val", type=float, default=0.2); t.add_argument("--seed", type=int, default=42)
    t.add_argument("--epochs", type=int, default=200); t.add_argument("--batch", type=int, default=32); t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--wd", type=float, default=1e-4); t.add_argument("--hidden", type=int, nargs="+", default=[64, 32]); t.add_argument("--dropout", type=float, default=0.15)
    t.add_argument("--patience", type=int, default=20); t.add_argument("--name"); t.add_argument("--device")
    p = sub.add_parser("predict"); p.add_argument("--csv", required=True); p.add_argument("--run", default="latest"); p.add_argument("--out"); p.add_argument("--device")
    s = sub.add_parser("selftest"); s.add_argument("--device")
    a = ap.parse_args()
    {"train": train, "predict": predict, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
