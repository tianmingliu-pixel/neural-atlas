"""GPT: autoregressive Transformer decoder (Radford et al., 2018/2019) with Hugging Face GPT-2.

  python models/zoo/gpt.py train    --text my_corpus.txt [--model gpt2] [--epochs 3]   (fine-tune pretrained GPT-2)
  python models/zoo/gpt.py generate --prompt "The firewall" [--run runs/zoo/gpt]
  python models/zoo/gpt.py selftest            (tiny GPT-2 trained from scratch on a bundled corpus, offline)

Text is tokenized, concatenated and cut into blocks; the model learns P(x_t | x_<t). Reports validation
perplexity before and after training, saves the model with save_pretrained, and samples continuations.
`gpt2` (124M parameters, ~500 MB) is downloaded on first use.
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE)]
import common as C  # noqa: E402
import _data as D  # noqa: E402


def blocks(tok, texts, size):
    ids = []
    for t in texts:
        ids += tok(t)["input_ids"] + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
    n = len(ids) // size
    if n < 2:
        sys.exit("Not enough text: need at least %d tokens, got %d" % (2 * size, len(ids)))
    return np.array(ids[: n * size], np.int64).reshape(n, size)


def perplexity(model, X, device, batch=16):
    import torch
    model.eval(); tot, cnt = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(X), batch):
            x = torch.as_tensor(X[s:s + batch], device=device)
            tot += float(model(input_ids=x, labels=x).loss) * len(x); cnt += len(x)
    return math.exp(tot / max(1, cnt))


def fit(model, tok, texts, args, device, dname, out):
    import torch
    X = blocks(tok, texts, args.block); rng = np.random.default_rng(args.seed); idx = rng.permutation(len(X))
    nv = max(1, len(X) // 10); Xv, Xt = X[idx[:nv]], X[idx[nv:]]
    model.to(device); opt = C.make_optimizer(model.parameters(), args.lr, 0.01, dname)
    p0 = perplexity(model, Xv, device); C.log("%d train / %d val blocks of %d tokens · val perplexity before %.2f · device %s" % (len(Xt), len(Xv), args.block, p0, dname))
    for ep in range(1, args.epochs + 1):
        model.train(); perm = rng.permutation(len(Xt)); tot = 0.0
        for s in range(0, len(perm), args.batch):
            x = torch.as_tensor(Xt[perm[s:s + args.batch]], device=device)
            loss = model(input_ids=x, labels=x).loss
            opt.zero_grad(); loss.backward(); C.clip_grads(model.parameters(), 1.0, dname); opt.step(); tot += float(loss) * len(x)
        C.log("epoch %d  train loss %.4f  val perplexity %.2f" % (ep, tot / len(Xt), perplexity(model, Xv, device)))
    p1 = perplexity(model, Xv, device)
    model.save_pretrained(out, state_dict=C.cpu_state(model)); tok.save_pretrained(out)
    C.save_json(os.path.join(out, "report.json"), {"val_perplexity_before": p0, "val_perplexity_after": p1})
    return p0, p1


def generate_text(model, tok, prompt, device, n=40, sample=True):
    import torch
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n, do_sample=sample, top_p=0.92, temperature=0.8, pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)


def train(args):
    from transformers import AutoTokenizer, GPT2LMHeadModel
    C.seed_all(args.seed)
    with open(args.text, encoding="utf-8", errors="replace") as f:
        texts = [p for p in f.read().split("\n\n") if p.strip()]
    tok = AutoTokenizer.from_pretrained(args.model); tok.pad_token = tok.pad_token or tok.eos_token
    device, dname = C.get_device(args.device)
    model = GPT2LMHeadModel.from_pretrained(args.model, **({"attn_implementation": "eager"} if dname.startswith("directml") else {}))
    out = D.out_dir("gpt"); fit(model, tok, texts, args, device, dname, out)
    print("\n" + generate_text(model, tok, texts[0][:40], device))


def generate(args):
    from transformers import AutoTokenizer, GPT2LMHeadModel
    run = args.run or D.out_dir("gpt"); device, _ = C.get_device(args.device)
    tok = AutoTokenizer.from_pretrained(run); model = GPT2LMHeadModel.from_pretrained(run).to(device).eval()
    for _ in range(args.n):
        print(generate_text(model, tok, args.prompt, device, args.max_new), "\n---")


def selftest(args):
    from transformers import GPT2Config, GPT2LMHeadModel
    C.seed_all(0)
    corpus = [s for v in D.text_corpus().values() for s in v] * 30
    tok = D.word_tokenizer(corpus)
    cfg = GPT2Config(vocab_size=len(tok), n_positions=64, n_embd=64, n_layer=2, n_head=2, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    try:
        cfg._attn_implementation = "eager"
    except Exception:
        pass
    model = GPT2LMHeadModel(cfg); device, dname = C.get_device(args.device)
    out = os.path.join(D.out_dir("gpt"), "selftest")
    p0, p1 = fit(model, tok, corpus, argparse.Namespace(block=32, epochs=6, batch=16, lr=3e-3, seed=0), device, dname, out)
    C.check(p1 < p0 / 4, "validation perplexity fell from %.1f to %.1f (> 4x)" % (p0, p1))
    txt = generate_text(model, tok, "the server", device, 8, sample=False); C.log("greedy continuation: %r" % txt)
    C.check(len(txt.split()) > 3, "model generates a continuation")
    C.log("GPT SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--text", required=True); t.add_argument("--model", default="gpt2"); t.add_argument("--block", type=int, default=128)
    t.add_argument("--epochs", type=int, default=3); t.add_argument("--batch", type=int, default=4); t.add_argument("--lr", type=float, default=5e-5); t.add_argument("--seed", type=int, default=0); t.add_argument("--device")
    g = sub.add_parser("generate"); g.add_argument("--prompt", required=True); g.add_argument("--run"); g.add_argument("--n", type=int, default=3); g.add_argument("--max-new", dest="max_new", type=int, default=60); g.add_argument("--device")
    s = sub.add_parser("selftest"); s.add_argument("--device")
    a = ap.parse_args(); {"train": train, "generate": generate, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
