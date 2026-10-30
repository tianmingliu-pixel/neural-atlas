"""Sequence-to-sequence with attention (Bahdanau et al., 2014): bidirectional GRU encoder + GRU decoder
with additive attention, trained to convert human-written dates to ISO format.

  python models/zoo/seq2seq.py train     [--n 20000] [--epochs 20]
  python models/zoo/seq2seq.py translate --text "3 March 2021"
  python models/zoo/seq2seq.py selftest

Writes an attention heat map (which input characters each output character looked at) to attention.png.
Compare with models/zoo/transformer.py, which solves the same task with nn.Transformer.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE)]
import common as C  # noqa: E402
import _data as D  # noqa: E402

SRC_LEN, TGT_LEN = 24, 12


def make_model(vocab_size, emb=64, hid=128):
    import torch
    import torch.nn as nn

    class Seq2Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
            self.enc = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
            self.bridge = nn.Linear(2 * hid, hid)
            self.W_enc, self.W_dec, self.v = nn.Linear(2 * hid, hid, bias=False), nn.Linear(hid, hid), nn.Linear(hid, 1, bias=False)
            self.dec = nn.GRUCell(emb + 2 * hid, hid)
            self.out = nn.Linear(hid + 2 * hid, vocab_size)

        def encode(self, src):
            H, h = self.enc(self.emb(src))                                   # H: [B,S,2h]
            return H, torch.tanh(self.bridge(torch.cat([h[0], h[1]], -1))), src != 0

        def step(self, y_prev, s, H, mask, keysH):
            e = self.v(torch.tanh(keysH + self.W_dec(s)[:, None])).squeeze(-1)  # additive attention scores [B,S]
            a = torch.softmax(e.masked_fill(~mask, -1e9), -1)
            ctx = (a[..., None] * H).sum(1)
            s = self.dec(torch.cat([self.emb(y_prev), ctx], -1), s)
            return self.out(torch.cat([s, ctx], -1)), s, a

        def forward(self, src, tgt_in, teacher=1.0):
            H, s, mask = self.encode(src); keysH = self.W_enc(H); y = tgt_in[:, 0]; outs = []
            for t in range(tgt_in.size(1)):
                logits, s, _ = self.step(y, s, H, mask, keysH); outs.append(logits)
                if t + 1 < tgt_in.size(1):
                    y = torch.where(torch.rand(len(y), device=y.device) < teacher, tgt_in[:, t + 1], logits.argmax(-1))
            return torch.stack(outs, 1)

        def greedy(self, src, bos, max_len):
            H, s, mask = self.encode(src); keysH = self.W_enc(H); y = torch.full((len(src),), bos, dtype=torch.long, device=src.device); out, att = [], []
            for _ in range(max_len):
                logits, s, a = self.step(y, s, H, mask, keysH); y = logits.argmax(-1); out.append(y); att.append(a)
            return torch.stack(out, 1), torch.stack(att, 1)
    return Seq2Seq()


def tensors(pairs, V):
    src = np.array([V.encode(a, SRC_LEN) for a, _ in pairs]); tgt = np.array([V.encode(b, TGT_LEN + 1, bos=True) for _, b in pairs])
    return src, tgt[:, :-1], tgt[:, 1:]


def exact_match(model, V, pairs, device):
    import torch
    src, _, _ = tensors(pairs, V); model.eval()
    with torch.no_grad():
        pred, _ = model.greedy(torch.as_tensor(src, device=device), V.BOS, TGT_LEN)
    pred = pred.cpu().numpy()
    return float(np.mean([V.decode(p) == b for p, (_, b) in zip(pred, pairs)]))


def train(args):
    import torch
    C.seed_all(args.seed)
    pairs = D.dates(args.n, args.seed); V = D.CharVocab([s for p in pairs for s in p] + [s for p in D.dates(3000, 99) for s in p])
    nv = len(pairs) // 10; val, tr = pairs[:nv], pairs[nv:]
    device, dname = C.get_device(args.device)
    model = make_model(len(V.itos)).to(device); opt = C.make_optimizer(model.parameters(), args.lr, 1e-5, dname)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    S, Ti, To = (torch.as_tensor(a) for a in tensors(tr, V))
    C.log("%d train / %d val pairs · vocab %d · device %s" % (len(tr), len(val), len(V.itos), dname))
    for ep in range(1, args.epochs + 1):
        model.train(); perm = torch.randperm(len(S)); tot = 0.0; teacher = max(0.3, 1.0 - ep / args.epochs)
        for s in range(0, len(S), args.batch):
            b = perm[s:s + args.batch]
            logits = model(S[b].to(device), Ti[b].to(device), teacher)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), To[b].to(device).reshape(-1))
            opt.zero_grad(); loss.backward(); C.clip_grads(model.parameters(), 1.0, dname); opt.step(); tot += float(loss) * len(b)
        em = exact_match(model, V, val, device)
        C.log("epoch %d  loss %.4f  teacher %.2f  val exact match %.3f" % (ep, tot / len(S), teacher, em))
    out = D.out_dir("seq2seq")
    torch.save({"state_dict": C.cpu_state(model), "itos": V.itos}, os.path.join(out, "model.pt"))
    attention_png(model, V, val[0][0], device, os.path.join(out, "attention.png"))
    C.save_json(os.path.join(out, "report.json"), {"val_exact_match": em, "examples": [(a, b) for a, b in val[:5]]})
    return em, model, V


def attention_png(model, V, text, device, path):
    import torch
    from PIL import Image, ImageDraw
    src = torch.as_tensor([V.encode(text, SRC_LEN)], device=device); model.eval()
    with torch.no_grad():
        pred, att = model.greedy(src, V.BOS, TGT_LEN)
    out_s = V.decode(pred[0].tolist()); A = att[0].cpu().numpy()[:len(out_s) + 1, :len(text) + 1]
    cell = 22; img = Image.new("RGB", ((A.shape[1] + 2) * cell, (A.shape[0] + 2) * cell), (8, 11, 18)); d = ImageDraw.Draw(img)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = int(255 * A[i, j] / (A[i].max() + 1e-9)); d.rectangle([(j + 2) * cell, (i + 2) * cell, (j + 3) * cell - 1, (i + 3) * cell - 1], fill=(v, int(v * .7), 40))
    for j, ch in enumerate(text + "$"):
        d.text(((j + 2) * cell + 6, cell + 4), ch, fill=(200, 210, 230))
    for i, ch in enumerate(out_s + "$"):
        d.text((cell + 4, (i + 2) * cell + 4), ch, fill=(200, 210, 230))
    img.save(path); C.log("attention map for %r -> %r saved to %s" % (text, out_s, path))


def translate(args):
    import torch
    device, _ = C.get_device(args.device)
    ck = C.load_ckpt(os.path.join(D.out_dir("seq2seq"), "model.pt"))
    V = D.CharVocab([]); V.itos = ck["itos"]; V.stoi = {c: i for i, c in enumerate(V.itos)}
    model = make_model(len(V.itos)); model.load_state_dict(ck["state_dict"]); model.to(device).eval()
    with torch.no_grad():
        pred, _ = model.greedy(torch.as_tensor([V.encode(args.text, SRC_LEN)], device=device), V.BOS, TGT_LEN)
    print(args.text, "->", V.decode(pred[0].tolist()))


def selftest(args):
    em, _, _ = train(argparse.Namespace(n=6000, epochs=14, batch=64, lr=3e-3, seed=0, device=args.device))
    C.check(em > 0.7, "date conversion exact match %.3f > 0.70" % em)
    C.log("SEQ2SEQ SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--n", type=int, default=20000); t.add_argument("--epochs", type=int, default=20); t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=2e-3); t.add_argument("--seed", type=int, default=0); t.add_argument("--device")
    p = sub.add_parser("translate"); p.add_argument("--text", required=True); p.add_argument("--device")
    s = sub.add_parser("selftest"); s.add_argument("--device")
    a = ap.parse_args(); {"train": train, "translate": translate, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
