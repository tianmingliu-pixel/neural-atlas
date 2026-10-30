"""Perceptron (Rosenblatt, 1958) with scikit-learn: y = sign(w·x + b).

  python models/zoo/perceptron.py train   [--csv data.csv --target col]   (default: bundled breast-cancer data)
  python models/zoo/perceptron.py predict --csv new.csv
  python models/zoo/perceptron.py selftest

Pipeline: median imputation -> standardization -> Perceptron. Reports 5-fold cross-validated accuracy,
a held-out confusion matrix and the learned weights (largest |w| = most influential features).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE)]
import common as C  # noqa: E402
import _data as D  # noqa: E402


def load(args):
    import pandas as pd
    if args.csv:
        df = pd.read_csv(args.csv)
        y = df[args.target].astype(str).values
        X = df.drop(columns=[args.target]).select_dtypes("number")
        return X.values.astype(np.float64), y, list(X.columns)
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    return d.data, np.array(d.target_names)[d.target], list(d.feature_names)


def train(args):
    import joblib
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Perceptron
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y, cols = load(args)
    classes = sorted(set(y)); yi = np.array([classes.index(v) for v in y])
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Perceptron(max_iter=1000, eta0=args.eta0, penalty=args.penalty, alpha=1e-4, random_state=args.seed))
    cv = cross_val_score(pipe, X, yi, cv=5, scoring="accuracy")
    C.log("5-fold CV accuracy %.3f ± %.3f" % (cv.mean(), cv.std()))
    tr, va = C.stratified_split(yi, 0.2, args.seed)
    pipe.fit(X[tr], yi[tr])
    rep = C.cls_report(yi[va], pipe.predict(X[va]), classes); C.print_cls_report(rep)
    w = pipe[-1].coef_
    top = sorted(zip(cols, (np.abs(w).max(0) if w.shape[0] > 1 else w[0])), key=lambda t: -abs(t[1]))[:8]
    C.log("largest weights:", ", ".join("%s=%.2f" % (c, v) for c, v in top))
    out = D.out_dir("perceptron"); joblib.dump({"pipe": pipe, "classes": classes, "cols": cols}, os.path.join(out, "model.joblib"))
    C.save_json(os.path.join(out, "report.json"), {"cv_accuracy": cv.tolist(), "validation": rep, "top_weights": top})
    C.log("saved to", out)
    return cv.mean(), rep


def predict(args):
    import joblib
    import pandas as pd
    m = joblib.load(os.path.join(D.out_dir("perceptron"), "model.joblib"))
    df = pd.read_csv(args.csv); pred = m["pipe"].predict(df[m["cols"]].values)
    df["prediction"] = [m["classes"][i] for i in pred]; print(df.head(20).to_string(index=False)); return df


def selftest(args):
    cv, rep = train(argparse.Namespace(csv=None, target=None, eta0=1.0, penalty=None, seed=0))
    C.check(cv > 0.9 and rep["accuracy"] > 0.9, "breast-cancer accuracy CV %.3f / held-out %.3f > 0.90" % (cv, rep["accuracy"]))
    C.log("PERCEPTRON SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--csv"); t.add_argument("--target"); t.add_argument("--eta0", type=float, default=1.0)
    t.add_argument("--penalty", choices=["l2", "l1", "elasticnet"]); t.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("predict"); p.add_argument("--csv", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args(); {"train": train, "predict": predict, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
