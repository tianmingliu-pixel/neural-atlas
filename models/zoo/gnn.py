"""Graph Convolutional Network (Kipf & Welling, 2017) with PyTorch Geometric.

  python models/zoo/gnn.py train   [--dataset karate|cora|citeseer] [--edges edges.csv --labels labels.csv]
  python models/zoo/gnn.py selftest           (Zachary's karate club, bundled with PyG, offline)

Node classification: each GCNConv layer averages a node's neighbours (with symmetric normalisation
D^-1/2 (A+I) D^-1/2) before a linear transform, so information flows along edges. `cora`/`citeseer`
are downloaded by PyG on first use. Your own graph: --edges (source,target per row) and --labels
(node,label per row; nodes without a label are predicted). Writes node embeddings and predictions.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE)]
import common as C  # noqa: E402
import _data as D  # noqa: E402


def make_gcn(n_in, hidden, n_out, dropout=0.5):
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    class GCN(nn.Module):
        def __init__(self):
            super().__init__(); self.c1, self.c2 = GCNConv(n_in, hidden), GCNConv(hidden, n_out); self.dropout = dropout

        def forward(self, x, edge_index, return_hidden=False):
            h = F.relu(self.c1(F.dropout(x, self.dropout, self.training), edge_index))
            out = self.c2(F.dropout(h, self.dropout, self.training), edge_index)
            return (out, h) if return_hidden else out
    return GCN()


def load(args):
    import torch
    if args.edges:
        from torch_geometric.data import Data
        with open(args.edges, encoding="utf-8") as f:
            E = [(r[0], r[1]) for r in csv.reader(f) if len(r) >= 2 and not r[0].startswith("#")]
        if E and not E[0][0].isdigit():
            E = E[1:]  # header
        names = sorted({n for e in E for n in e}); idx = {n: i for i, n in enumerate(names)}
        lab = {}
        if args.labels:
            with open(args.labels, encoding="utf-8") as f:
                for r in csv.reader(f):
                    if len(r) >= 2 and r[0] in idx:
                        lab[r[0]] = r[1]
        classes = sorted(set(lab.values())); y = torch.full((len(names),), -1, dtype=torch.long)
        for n, l in lab.items():
            y[idx[n]] = classes.index(l)
        ei = torch.tensor([[idx[a] for a, b in E] + [idx[b] for a, b in E], [idx[b] for a, b in E] + [idx[a] for a, b in E]])
        deg = torch.bincount(ei[0], minlength=len(names)).float()[:, None]
        x = torch.cat([torch.eye(len(names)), torch.log1p(deg)], 1) if len(names) <= 5000 else torch.log1p(deg)
        known = (y >= 0).nonzero().flatten(); perm = known[torch.randperm(len(known))]; nv = max(1, len(known) // 5)
        tm, vm = torch.zeros(len(names), dtype=torch.bool), torch.zeros(len(names), dtype=torch.bool); tm[perm[nv:]] = True; vm[perm[:nv]] = True
        return Data(x=x, edge_index=ei, y=y, train_mask=tm, val_mask=vm), classes, names
    if args.dataset == "karate":
        from torch_geometric.datasets import KarateClub
        d = KarateClub()[0]; n = d.num_nodes
        vm = ~d.train_mask
        return d.__class__(x=d.x, edge_index=d.edge_index, y=d.y, train_mask=d.train_mask, val_mask=vm), [str(i) for i in range(int(d.y.max()) + 1)], [str(i) for i in range(n)]
    from torch_geometric.datasets import Planetoid
    ds = Planetoid(os.path.join(D.out_dir("data"), "planetoid"), {"cora": "Cora", "citeseer": "CiteSeer"}[args.dataset]); d = ds[0]
    return d, [str(i) for i in range(ds.num_classes)], [str(i) for i in range(d.num_nodes)]


def train(args):
    import torch
    C.seed_all(args.seed)
    data, classes, names = load(args)
    device, dname = C.get_device(args.device)
    if dname.startswith("directml"):
        device, dname = torch.device("cpu"), "cpu (scatter ops are not supported on DirectML)"
    data = data.to(device); model = make_gcn(data.num_features, args.hidden, len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4); loss_fn = torch.nn.CrossEntropyLoss()
    C.log("GCN · %d nodes · %d edges · %d features · %d classes · %d labelled for training · device %s" % (data.num_nodes, data.edge_index.shape[1], data.num_features, len(classes), int(data.train_mask.sum()), dname))
    best, best_state = -1, None
    for ep in range(1, args.epochs + 1):
        model.train(); out = model(data.x, data.edge_index)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask]); opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(1)
        acc = float((pred[data.val_mask] == data.y[data.val_mask]).float().mean())
        if acc >= best:
            best, best_state = acc, C.cpu_state(model)
        if ep == 1 or ep % max(1, args.epochs // 8) == 0:
            C.log("epoch %d  loss %.4f  val acc %.3f" % (ep, float(loss), acc))
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        logits, hidden = model(data.x, data.edge_index, return_hidden=True)
    pred = logits.argmax(1).cpu().numpy(); emb = hidden.cpu().numpy()
    out = D.out_dir("gnn")
    with open(os.path.join(out, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["node", "predicted", "known_label"] + ["h%d" % i for i in range(emb.shape[1])])
        yk = data.y.cpu().numpy()
        for i, n in enumerate(names):
            w.writerow([n, classes[pred[i]], classes[yk[i]] if yk[i] >= 0 else ""] + [round(float(v), 4) for v in emb[i]])
    torch.save({"state_dict": best_state, "n_in": data.num_features, "hidden": args.hidden, "classes": classes}, os.path.join(out, "gcn.pt"))
    C.save_json(os.path.join(out, "report.json"), {"best_val_acc": best, "dataset": args.dataset if not args.edges else "custom"})
    C.log("best val accuracy %.3f · predictions + embeddings -> %s" % (best, os.path.join(out, "predictions.csv")))
    return best


def selftest(args):
    acc = train(argparse.Namespace(dataset="karate", edges=None, labels=None, hidden=16, epochs=200, lr=0.01, seed=0, device=args.device))
    C.check(acc > 0.6, "karate club: %.3f of unlabelled members classified correctly from 4 labels (> 0.60, chance 0.25)" % acc)
    C.log("GNN SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--dataset", choices=["karate", "cora", "citeseer"], default="cora"); t.add_argument("--edges"); t.add_argument("--labels")
    t.add_argument("--hidden", type=int, default=16); t.add_argument("--epochs", type=int, default=200); t.add_argument("--lr", type=float, default=0.01); t.add_argument("--seed", type=int, default=0); t.add_argument("--device")
    s = sub.add_parser("selftest"); s.add_argument("--device")
    a = ap.parse_args(); {"train": train, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
