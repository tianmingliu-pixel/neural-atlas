"""Kolmogorov-Arnold Network with the official pykan library (Liu et al., 2024).

  python models/zoo/kan_net.py train   [--func "exp(sin(pi*x0) + x1**2)"] [--width 2 5 1]
  python models/zoo/kan_net.py selftest

KANs put learnable spline functions on the edges. After training, pruning and `auto_symbolic`, the
network can be read back as a closed-form formula — the main reason to use a KAN.
Install: pip install pykan  (pulls sympy, matplotlib, tqdm, pyyaml)
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE)]
import common as C  # noqa: E402
import _data as D  # noqa: E402

SAFE = {"exp", "sin", "cos", "tanh", "log", "sqrt", "abs", "pi"}


def target_fn(expr, n_var):
    import torch
    names = set(__import__("re").findall(r"[A-Za-z_]+\d*", expr)) - SAFE - {"x%d" % i for i in range(n_var)}
    if names:
        sys.exit("Unsupported names in --func: %s (use x0..x%d and %s)" % (sorted(names), n_var - 1, sorted(SAFE)))
    env = {k: getattr(torch, k) for k in SAFE if k != "pi"}; env["pi"] = torch.pi
    return lambda x: eval(expr, {"__builtins__": {}}, dict(env, **{"x%d" % i: x[:, [i]] for i in range(n_var)}))


def make_kan(width, grid, k, seed, device):
    from kan import KAN
    try:
        return KAN(width=width, grid=grid, k=k, seed=seed, device=device, auto_save=False)
    except TypeError:  # older pykan without auto_save/device
        return KAN(width=width, grid=grid, k=k, seed=seed)


def fit(model, dataset, steps, lamb):
    f = getattr(model, "fit", None)
    if f is None:  # pykan < 0.2 called it `train`
        return model.train(dataset, opt="LBFGS", steps=steps, lamb=lamb)
    return f(dataset, opt="LBFGS", steps=steps, lamb=lamb)


def rmse(model, dataset):
    import torch
    with torch.no_grad():
        return float(torch.sqrt(((model(dataset["test_input"]) - dataset["test_label"]) ** 2).mean()))


def train(args):
    import torch
    from kan import create_dataset
    C.seed_all(args.seed)
    device = "cpu"  # LBFGS + spline grids are small; CPU is the reliable choice here
    n_var = args.width[0]
    f = target_fn(args.func, n_var)
    dataset = create_dataset(f, n_var=n_var, train_num=args.n, test_num=args.n, device=device)
    model = make_kan(args.width, args.grid, args.k, args.seed, device)
    fit(model, dataset, args.steps, args.lamb)
    r1 = rmse(model, dataset); C.log("after training: test RMSE %.5f" % r1)
    model = model.prune(); fit(model, dataset, args.steps, 0.0)
    r2 = rmse(model, dataset); C.log("after pruning: test RMSE %.5f" % r2)
    model.auto_symbolic(lib=args.lib)
    fit(model, dataset, max(10, args.steps // 3), 0.0)
    r3 = rmse(model, dataset)
    formula = str(model.symbolic_formula()[0][0])
    C.log("symbolic formula: %s  (test RMSE %.5f)" % (formula, r3))
    out = D.out_dir("kan")
    torch.save(model.state_dict(), os.path.join(out, "kan_state.pt"))
    C.save_json(os.path.join(out, "report.json"), {"func": args.func, "width": args.width, "rmse_trained": r1, "rmse_pruned": r2, "rmse_symbolic": r3, "formula": formula})
    C.log("saved to", out)
    return r1, r3, formula


def selftest(args):
    r1, r3, formula = train(argparse.Namespace(func="exp(sin(pi*x0) + x1**2)", width=[2, 5, 1], grid=5, k=3, n=1000, steps=50, lamb=0.001, seed=1,
                                               lib=["x", "x^2", "exp", "sin"]))
    C.check(r1 < 0.1, "spline KAN fits the target (RMSE %.4f < 0.1)" % r1)
    C.check("sin" in formula, "recovered formula contains the sin term: %s" % formula)
    if "exp" not in formula:
        C.log("note: exp was not recovered this time (depends on pruning); the target is exp(sin(pi*x0) + x1**2)")
    C.log("KAN SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--func", default="exp(sin(pi*x0) + x1**2)"); t.add_argument("--width", type=int, nargs="+", default=[2, 5, 1])
    t.add_argument("--grid", type=int, default=5); t.add_argument("--k", type=int, default=3); t.add_argument("--n", type=int, default=1000)
    t.add_argument("--steps", type=int, default=50); t.add_argument("--lamb", type=float, default=0.001); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--lib", nargs="+", default=["x", "x^2", "x^3", "exp", "sin", "tanh", "log", "sqrt"])
    sub.add_parser("selftest")
    a = ap.parse_args(); {"train": train, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
