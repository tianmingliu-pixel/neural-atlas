"""Transformer encoder-decoder (Vaswani et al., 2017) with torch.nn.Transformer.

  python models/zoo/transformer.py train     [--n 20000] [--epochs 15]
  python models/zoo/transformer.py translate --text "aug 28 2000"
  python models/zoo/transformer.py selftest

Same date-conversion task as seq2seq.py so the two can be compared directly. Uses sinusoidal
positional encoding, padding masks, a causal decoder mask, label smoothing, warmup + cosine LR
and greedy decoding.
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE)]
import common as C  # noqa: E402
import _data as D  # noqa: E402

SRC_LEN, TGT_LEN = 24, 12


def make_model(vocab, d=128, heads=4, layers=3, ff=256, drop=0.1):
    import torch
    import torch.nn as nn

    class PosEnc(nn.Module):
        def __init__(self, n=64):
            super().__init__()
            pe = torch.zeros(n, d); pos = torch.arange(n)[:, None]; div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div); self.register_buffer("pe", pe)

        def forward(self, x):
            return x + self.pe[: x.size(1)]

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, d, padding_idx=0); self.pos = PosEnc(); self.drop = nn.Dropout(drop)
            self.tf = nn.Transformer(d_model=d, nhead=heads, num_encoder_layers=layers, num_decoder_layers=layers, dim_feedforward=ff, dropout=drop, batch_first=True)
            self.out = nn.Linear(d, vocab)

        def embed(self, x):
            return self.drop(self.pos(self.emb(x) * math.sqrt(d)))

        def forward(self, src, tgt_in):
            causal = torch.triu(torch.ones(tgt_in.size(1), tgt_in.size(1), dtype=torch.bool, device=src.device), 1)
            h = self.tf(self.embed(src), self.embed(tgt_in), tgt_mask=causal, src_key_padding_mask=src == 0,
                        tgt_key_padding_mask=tgt_in == 0, memory_key_padding_mask=src == 0)
            return self.out(h)

        def greedy(self, src, bos, max_len):
            mem = self.tf.encoder(self.embed(src), src_key_padding_mask=src == 0)
            ys = torch.full((len(src), 1), bos, dtype=torch.long, device=src.device)
            for _ in range(max_len):
                causal = torch.triu(torch.ones(ys.size(1), ys.size(1), dtype=torch.bool, device=src.device), 1)
                h = self.tf.decoder(self.embed(ys), mem, tgt_mask=causal, memory_key_padding_mask=src == 0)
                ys = torch.cat([ys, self.out(h[:, -1]).argmax(-1, keepdim=True)], 1)
            return ys[:, 1:]
    return Model()


def tensors(pairs, V):
    src = np.array([V.encode(a, SRC_LEN) for a, _ in pairs]); tgt = np.array([V.encode(b, TGT_LEN + 1, bos=True) for _, b in pairs])
    return src, tgt[:, :-1], tgt[:, 1:]


def exact_match(model, V, pairs, device):
    import torch
    src, _, _ = tensors(pairs, V); model.eval()
    with torch.no_grad():
        pred = model.greedy(torch.as_tensor(src, device=device), V.BOS, TGT_LEN).cpu().numpy()
    return float(np.mean([V.decode(p) == b for p, (_, b) in zip(pred, pairs)]))


def train(args):
    import torch
    C.seed_all(args.seed)
    pairs = D.dates(args.n, args.seed); V = D.CharVocab([s for p in pairs for s in p] + [s for p in D.dates(3000, 99) for s in p])
    nv = len(pairs) // 10; val, tr = pairs[:nv], pairs[nv:]
    device, dname = C.get_device(args.device); C.safe_attention(dname)
    model = make_model(len(V.itos)).to(device); opt = C.make_optimizer(model.parameters(), args.lr, 0.01, dname)
    steps = args.epochs * math.ceil(len(tr) / args.batch); warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.05)
    S, Ti, To = (torch.as_tensor(a) for a in tensors(tr, V))
    C.log("%d train / %d val · %d params · device %s" % (len(tr), len(val), sum(p.numel() for p in model.parameters()), dname))
    for ep in range(1, args.epochs + 1):
        model.train(); perm = torch.randperm(len(S)); tot = 0.0
        for s in range(0, len(S), args.batch):
            b = perm[s:s + args.batch]
            logits = model(S[b].to(device), Ti[b].to(device))
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), To[b].to(device).reshape(-1))
            opt.zero_grad(); loss.backward(); C.clip_grads(model.parameters(), 1.0, dname); opt.step(); sched.step(); tot += float(loss) * len(b)
        em = exact_match(model, V, val, device)
        C.log("epoch %d  loss %.4f  val exact match %.3f" % (ep, tot / len(S), em))
    out = D.out_dir("transformer")
    torch.save({"state_dict": C.cpu_state(model), "itos": V.itos}, os.path.join(out, "model.pt"))
    C.save_json(os.path.join(out, "report.json"), {"val_exact_match": em})
    return em


def translate(args):
    import torch
    device, dname = C.get_device(args.device); C.safe_attention(dname)
    ck = C.load_ckpt(os.path.join(D.out_dir("transformer"), "model.pt"))
    V = D.CharVocab([]); V.itos = ck["itos"]; V.stoi = {c: i for i, c in enumerate(V.itos)}
    model = make_model(len(V.itos)); model.load_state_dict(ck["state_dict"]); model.to(device).eval()
    with torch.no_grad():
        pred = model.greedy(torch.as_tensor([V.encode(args.text, SRC_LEN)], device=device), V.BOS, TGT_LEN)
    print(args.text, "->", V.decode(pred[0].tolist()))


def selftest(args):
    em = train(argparse.Namespace(n=6000, epochs=14, batch=64, lr=1e-3, seed=0, device=args.device))
    C.check(em > 0.7, "date conversion exact match %.3f > 0.70" % em)
    C.log("TRANSFORMER SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--n", type=int, default=20000); t.add_argument("--epochs", type=int, default=15); t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--seed", type=int, default=0); t.add_argument("--device")
    p = sub.add_parser("translate"); p.add_argument("--text", required=True); p.add_argument("--device")
    s = sub.add_parser("selftest"); s.add_argument("--device")
    a = ap.parse_args(); {"train": train, "translate": translate, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
