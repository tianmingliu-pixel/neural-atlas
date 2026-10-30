"""Mixture of Experts with the Hugging Face Mixtral implementation (sparse top-2 routing).

  python models/zoo/moe.py train    [--text corpus.txt] [--experts 4] [--top-k 2]
  python models/zoo/moe.py selftest            (bundled 3-domain corpus, offline)

The full Mixtral-8x7B needs ~90 GB of GPU memory, so this trains a small Mixtral from scratch with the
same code path (MixtralForCausalLM): every token is routed to the top-k of N expert MLPs, and the
load-balancing auxiliary loss keeps experts used evenly. After training, the script reports which experts
the router picks for each text domain — whether experts specialise.
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE)]
import common as C  # noqa: E402
import _data as D  # noqa: E402


def make_model(tok, args, dname):
    from transformers import MixtralConfig, MixtralForCausalLM
    cfg = MixtralConfig(vocab_size=len(tok), hidden_size=args.hidden, intermediate_size=args.hidden * 2, num_hidden_layers=args.layers, num_attention_heads=4,
                        num_key_value_heads=2, num_local_experts=args.experts, num_experts_per_tok=args.top_k, max_position_embeddings=128,
                        router_aux_loss_coef=0.02, output_router_logits=True, sliding_window=None,
                        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    try:
        cfg._attn_implementation = "eager"
    except Exception:
        pass
    return MixtralForCausalLM(cfg)


def encode(tok, texts, size):
    X = np.full((len(texts), size), tok.pad_token_id, np.int64)
    for i, t in enumerate(texts):
        ids = ([tok.bos_token_id] + tok(t)["input_ids"] + [tok.eos_token_id])[:size]; X[i, :len(ids)] = ids
    return X


def expert_usage(model, tok, domains, device, size):
    """Share of top-1 routing decisions per expert, per domain, averaged over layers."""
    import torch
    model.eval(); usage = {}
    with torch.no_grad():
        for name, texts in domains.items():
            X = torch.as_tensor(encode(tok, texts, size), device=device); mask = (X != tok.pad_token_id).reshape(-1).cpu().numpy()
            rl = model(input_ids=X, attention_mask=(X != tok.pad_token_id).long(), output_router_logits=True).router_logits
            cnt = np.zeros(model.config.num_local_experts)
            for layer_logits in rl:
                top = layer_logits.argmax(-1).cpu().numpy()[mask]
                cnt += np.bincount(top, minlength=len(cnt))
            usage[name] = (cnt / cnt.sum()).round(3).tolist()
    return usage


def fit(model, tok, texts, domains, args, device, dname):
    import torch
    X = encode(tok, texts, args.size); rng = np.random.default_rng(args.seed); idx = rng.permutation(len(X)); nv = max(1, len(X) // 10)
    Xv, Xt = X[idx[:nv]], X[idx[nv:]]
    model.to(device); opt = C.make_optimizer(model.parameters(), args.lr, 0.01, dname)
    def val_ppl():
        model.eval()
        with torch.no_grad():
            x = torch.as_tensor(Xv, device=device); lab = x.clone(); lab[lab == tok.pad_token_id] = -100
            return math.exp(float(model(input_ids=x, attention_mask=(x != tok.pad_token_id).long(), labels=lab, output_router_logits=False).loss))
    p0 = val_ppl()
    C.log("Mixtral-style MoE · %d experts, top-%d · %d params · val perplexity %.2f · device %s" % (args.experts, args.top_k, sum(p.numel() for p in model.parameters()), p0, dname))
    for ep in range(1, args.epochs + 1):
        model.train(); perm = rng.permutation(len(Xt)); tot = aux = 0.0
        for s in range(0, len(perm), args.batch):
            x = torch.as_tensor(Xt[perm[s:s + args.batch]], device=device); lab = x.clone(); lab[lab == tok.pad_token_id] = -100
            out = model(input_ids=x, attention_mask=(x != tok.pad_token_id).long(), labels=lab, output_router_logits=True)
            opt.zero_grad(); out.loss.backward(); C.clip_grads(model.parameters(), 1.0, dname); opt.step()
            tot += float(out.loss) * len(x); aux += float(out.aux_loss) * len(x) if getattr(out, "aux_loss", None) is not None else 0.0
        C.log("epoch %d  loss %.4f (incl. load-balancing aux %.4f)  val perplexity %.2f" % (ep, tot / len(Xt), aux / len(Xt), val_ppl()))
    p1 = val_ppl(); usage = expert_usage(model, tok, domains, device, args.size)
    for name, u in usage.items():
        C.log("expert share for %-8s %s" % (name, "  ".join("E%d %.0f%%" % (i, 100 * v) for i, v in enumerate(u))))
    return p0, p1, usage


def train(args):
    domains = D.text_corpus() if not args.text else {"text": [l.strip() for l in open(args.text, encoding="utf-8", errors="replace") if l.strip()]}
    texts = [t for v in domains.values() for t in v] * (30 if not args.text else 1)
    tok = D.word_tokenizer(texts); device, dname = C.get_device(args.device)
    model = make_model(tok, args, dname)
    p0, p1, usage = fit(model, tok, texts, domains, args, device, dname)
    out = D.out_dir("moe"); model.save_pretrained(out, state_dict=C.cpu_state(model)); tok.save_pretrained(out)
    C.save_json(os.path.join(out, "report.json"), {"val_perplexity_before": p0, "val_perplexity_after": p1, "expert_usage": usage})
    return p0, p1, usage


def selftest(args):
    C.seed_all(0)
    p0, p1, usage = train(argparse.Namespace(text=None, experts=4, top_k=2, hidden=64, layers=2, size=24, epochs=8, batch=16, lr=3e-3, seed=0, device=args.device))
    C.check(p1 < p0 / 4, "perplexity fell from %.1f to %.1f (> 4x)" % (p0, p1))
    C.check(all(abs(sum(u) - 1) < 0.01 for u in usage.values()), "router statistics collected for every domain")
    C.log("MOE SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--text"); t.add_argument("--experts", type=int, default=4); t.add_argument("--top-k", dest="top_k", type=int, default=2)
    t.add_argument("--hidden", type=int, default=128); t.add_argument("--layers", type=int, default=4); t.add_argument("--size", type=int, default=64)
    t.add_argument("--epochs", type=int, default=10); t.add_argument("--batch", type=int, default=16); t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--seed", type=int, default=0); t.add_argument("--device")
    s = sub.add_parser("selftest"); s.add_argument("--device")
    a = ap.parse_args(); {"train": train, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
