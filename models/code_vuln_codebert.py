"""Code vulnerability classifier: fine-tune CodeBERT (Hugging Face transformers) on your labeled code.

  python models/code_vuln_codebert.py train   --data semgrep.json [--data more.jsonl] [--safe-dir path\\to\\reviewed\\code]
  python models/code_vuln_codebert.py predict --file app.py   (or --code "...")   [--run latest]
  python models/code_vuln_codebert.py selftest              (offline: tiny BERT + benign language-ID data)

Data formats (mix freely):
  * Semgrep JSON (`semgrep --json`): each finding's code lines, labeled by rule id / CWE into
    sqli, cmdi, path, xss, deser, secret, crypto (or the rule's last id segment when no category matches)
  * JSONL / JSON array with {"code": ..., "label": ...}
  * CSV with columns code,label
  * --safe-dir: source files you reviewed as clean, cut into chunks and labeled "safe"
Semgrep findings contain only flagged code, so add --safe-dir (or "safe" rows) for the model to learn the difference.
Default base model: microsoft/codebert-base (~500 MB download on first use).
"""
import argparse
import csv
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

TASK = "code"
CODE_EXT = (".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".php", ".go", ".rb", ".cs", ".c", ".cpp", ".h", ".kt", ".scala", ".rs", ".swift")


# ---------------- data ----------------
def semgrep_label(text):
    s = text.lower()
    rules = [("sqli", r"sql|cwe-89"), ("cmdi", r"command|subprocess|shell|cwe-78"), ("path", r"path.?travers|cwe-22|file.?inclusion"),
             ("xss", r"xss|cross.?site|cwe-79"), ("deser", r"deserial|cwe-502|cwe-94|code.?injection"), ("secret", r"secret|hardcoded|credential|cwe-798"),
             ("crypto", r"crypto|hash|cipher|random|cwe-327|cwe-328|cwe-338")]
    for lab, rx in rules:
        if re.search(rx, s):
            return lab
    return None


def load_items(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read().strip()
    items = []
    if text.startswith("{") or text.startswith("["):
        try:
            j = json.loads(text)
        except json.JSONDecodeError:
            j = None
        if isinstance(j, dict) and isinstance(j.get("results"), list):
            for r in j["results"]:
                ex = r.get("extra") or {}
                code = ex.get("lines") or ""
                if not code.strip() or "requires login" in code:
                    continue
                meta = ex.get("metadata") or {}
                lab = semgrep_label(" ".join([str(r.get("check_id", "")), str(ex.get("message", "")), json.dumps(meta.get("cwe", ""))])) or str(r.get("check_id", "finding")).split(".")[-1]
                items.append({"code": code, "label": lab, "src": "semgrep", "ref": "%s:%s" % (r.get("path", ""), (r.get("start") or {}).get("line", ""))})
            return items
        if isinstance(j, list):
            return [{"code": str(o["code"]), "label": str(o["label"]), "src": "json"} for o in j if isinstance(o, dict) and o.get("code") and o.get("label")]
    lines = [l for l in text.splitlines() if l.strip()]
    if lines and all(l.lstrip().startswith("{") for l in lines):
        for l in lines:
            try:
                o = json.loads(l)
                if o.get("code") and o.get("label"):
                    items.append({"code": str(o["code"]), "label": str(o["label"]), "src": "jsonl"})
            except json.JSONDecodeError:
                pass
        return items
    import io
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    if rows and "code" in rows[0] and "label" in rows[0]:
        return [{"code": r["code"], "label": r["label"], "src": "csv"} for r in rows if r.get("code") and r.get("label")]
    sys.exit("Unrecognized data file: %s (expected Semgrep JSON, JSON/JSONL with code+label, or CSV with code,label)" % path)


def safe_chunks(folder, lines_per_chunk=25, max_chunks=3000):
    out = []
    for root, _, files in os.walk(folder):
        parts = set(os.path.normpath(root).split(os.sep))
        if parts & {".git", "node_modules", "__pycache__", "venv", ".venv"}:
            continue
        for fn in files:
            if not fn.endswith(CODE_EXT):
                continue
            try:
                with open(os.path.join(root, fn), encoding="utf-8", errors="replace") as f:
                    src = f.read().splitlines()
            except OSError:
                continue
            for i in range(0, len(src), lines_per_chunk):
                chunk = "\n".join(src[i:i + lines_per_chunk]).strip()
                if len(chunk) > 40:
                    out.append({"code": chunk, "label": "safe", "src": "safe-dir", "ref": "%s:%d" % (os.path.join(root, fn), i + 1)})
                if len(out) >= max_chunks:
                    return out
    return out


# ---------------- model helpers ----------------
def encode(tok, codes, max_len):
    return tok(codes, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")


def evaluate(model, tok, codes, y, labels, device, max_len, batch=32):
    import torch
    model.eval(); preds = []
    with torch.no_grad():
        for s in range(0, len(codes), batch):
            enc = encode(tok, codes[s:s + batch], max_len)
            out = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)).logits
            preds.append(out.argmax(-1).cpu().numpy())
    pred = np.concatenate(preds) if preds else np.zeros(0, int)
    return C.cls_report(np.asarray(y), pred, labels)


def fit(model, tok, items, labels, args, device, dname, out_dir):
    import torch
    from transformers import get_linear_schedule_with_warmup
    y = np.array([labels.index(i["label"]) for i in items]); codes = [i["code"] for i in items]
    tr, va = C.stratified_split(y, args.val, args.seed)
    C.log("%d train / %d val · %d classes · device %s" % (len(tr), len(va), len(labels), dname))
    model.to(device)
    opt = C.make_optimizer(model.parameters(), args.lr, args.wd, dname)
    steps = max(1, (len(tr) + args.batch - 1) // args.batch * args.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(steps * 0.06), steps)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.as_tensor(C.class_weights(y[tr], len(labels)), device=device))
    stopper, hist, best = C.EarlyStopper(args.patience, "max"), [], None
    rng = np.random.default_rng(args.seed)
    for ep in range(1, args.epochs + 1):
        model.train(); perm = tr[rng.permutation(len(tr))]; tot = 0.0
        for s in range(0, len(perm), args.batch):
            b = perm[s:s + args.batch]
            enc = encode(tok, [codes[i] for i in b], args.max_len)
            logits = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)).logits
            loss = loss_fn(logits, torch.as_tensor(y[b], device=device))
            opt.zero_grad(); loss.backward(); C.clip_grads(model.parameters(), 1.0, dname); opt.step(); sched.step(); tot += float(loss) * len(b)
        rep = evaluate(model, tok, [codes[i] for i in va], y[va], labels, device, args.max_len)
        hist.append({"epoch": ep, "loss": tot / len(tr), "val_macro_f1": rep["macro_f1"], "val_acc": rep["accuracy"]})
        C.log("epoch %d  loss %.4f  val acc %.3f  macro F1 %.3f" % (ep, tot / len(tr), rep["accuracy"], rep["macro_f1"]))
        if stopper.step(rep["macro_f1"]):
            best = rep; model.save_pretrained(out_dir, state_dict=C.cpu_state(model)); tok.save_pretrained(out_dir)
        if stopper.stop:
            C.log("early stop"); break
    C.print_cls_report(best)
    errors = []
    return best, hist, errors


def train(args):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    C.seed_all(args.seed)
    items = []
    for p in args.data or []:
        got = load_items(p); C.log("loaded %d items from %s" % (len(got), p)); items += got
    if args.safe_dir:
        got = safe_chunks(args.safe_dir); C.log("loaded %d safe chunks from %s" % (len(got), args.safe_dir)); items += got
    counts = {}
    for i in items:
        counts[i["label"]] = counts.get(i["label"], 0) + 1
    keep = {k for k, v in counts.items() if v >= args.min_per_label}
    dropped = {k: v for k, v in counts.items() if k not in keep}
    if dropped:
        C.log("dropping labels with fewer than %d samples: %s" % (args.min_per_label, dropped))
    items = [i for i in items if i["label"] in keep]
    labels = sorted(keep)
    if len(labels) < 2:
        sys.exit("Need at least 2 labels with >= %d samples each. Current counts: %s. Add --safe-dir or more data." % (args.min_per_label, counts))
    C.log("label counts:", {k: counts[k] for k in labels})
    device, dname = C.get_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    extra = {"attn_implementation": "eager"} if dname.startswith("directml") else {}
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)}, **extra)
    out = C.new_run_dir(TASK, args.name)
    best, hist, _ = fit(model, tok, items, labels, args, device, dname, out)
    C.save_json(os.path.join(out, "labels.json"), labels)
    C.save_json(os.path.join(out, "report.json"), {"base_model": args.model, "validation": best, "history": hist, "label_counts": {k: counts[k] for k in labels}, "device": dname, "max_len": args.max_len})
    C.log("saved to", out)
    return out, best


def saliency(model, tok, code, target, device, max_len):
    """Gradient x input on the input embeddings, mapped back to character spans of `code`."""
    import torch
    enc = tok(code, truncation=True, max_length=max_len, return_offsets_mapping=True, return_tensors="pt")
    ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    model.zero_grad(); model(inputs_embeds=emb, attention_mask=mask).logits[0, target].backward()
    score = (emb.grad * emb).sum(-1)[0].clamp(min=0).detach().cpu().numpy()
    spans = enc["offset_mapping"][0].tolist()
    best = sorted(((float(sc), a, b) for sc, (a, b) in zip(score, spans) if b > a), reverse=True)[:10]
    mx = best[0][0] if best and best[0][0] > 0 else 1.0
    return [{"text": code[a:b], "start": a, "end": b, "weight": round(sc / mx, 3)} for sc, a, b in best]


def predict(args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    run = C.resolve_run(TASK, args.run)
    rep = C.load_json(os.path.join(run, "report.json")); labels = C.load_json(os.path.join(run, "labels.json"))
    device, dname = C.get_device(args.device)
    extra = {"attn_implementation": "eager"} if dname.startswith("directml") else {}
    tok = AutoTokenizer.from_pretrained(run); model = AutoModelForSequenceClassification.from_pretrained(run, **extra).to(device).eval()
    if args.code:
        chunks = [("input", args.code)]
    else:
        with open(args.file, encoding="utf-8", errors="replace") as f:
            src = f.read().splitlines()
        chunks = [("%s:%d" % (args.file, i + 1), "\n".join(src[i:i + 25])) for i in range(0, len(src), 25) if "\n".join(src[i:i + 25]).strip()]
    results = []
    for ref, code in chunks:
        enc = tok(code, truncation=True, max_length=rep.get("max_len", 256), return_tensors="pt")
        with torch.no_grad():
            p = torch.softmax(model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)).logits[0], -1).cpu().numpy()
        k = int(p.argmax())
        r = {"ref": ref, "label": labels[k], "confidence": round(float(p[k]), 4), "probs": {labels[i]: round(float(v), 4) for i, v in enumerate(p)}}
        if labels[k] != "safe" or args.explain_all:
            r["evidence"] = saliency(model, tok, code, k, device, rep.get("max_len", 256))
        results.append(r)
        flag = "" if labels[k] == "safe" else "  <-- review"
        print("%-28s %-10s %.3f%s" % (ref[-28:], labels[k], p[k], flag))
        if "evidence" in r:
            print("    evidence:", ", ".join("%s(%.2f)" % (e["text"].strip()[:24], e["weight"]) for e in r["evidence"][:6]))
    out = args.out or os.path.join(run, "predictions.json"); C.save_json(out, results); C.log("wrote", out)
    return results


# ---------------- self-test: tiny offline BERT, benign language-ID data ----------------
LANG_TPL = {
    "python": ["def {f}({a}, {b}):\n    return {a} + {b}", "for {a} in {l}:\n    print({a})", "class {C}:\n    def __init__(self, {a}):\n        self.{a} = {a}", "{l} = [{a} * 2 for {a} in range(10)]",
               "with open('{n}.txt') as fh:\n    {a} = fh.read()", "if {a} is None:\n    raise ValueError('missing {a}')"],
    "javascript": ["function {f}({a}, {b}) {{\n  return {a} + {b};\n}}", "const {l} = [1, 2, 3].map({a} => {a} * 2);", "{l}.forEach(({a}) => console.log({a}));",
                   "class {C} {{\n  constructor({a}) {{ this.{a} = {a}; }}\n}}", "async function {f}() {{\n  const res = await fetch('/api/{n}');\n  return res.json();\n}}"],
    "java": ["public int {f}(int {a}, int {b}) {{\n    return {a} + {b};\n}}", "for (int i = 0; i < {a}.length; i++) {{\n    System.out.println({a}[i]);\n}}",
             "public class {C} {{\n    private String {a};\n}}", "List<String> {l} = new ArrayList<>();", "Map<String, Integer> {a} = new HashMap<>();"],
    "sql": ["SELECT {a}, {b} FROM {n} WHERE {a} > 10;", "INSERT INTO {n} ({a}, {b}) VALUES (1, 2);", "UPDATE {n} SET {a} = 0 WHERE {b} IS NULL;",
            "SELECT COUNT(*) FROM {n} GROUP BY {a} ORDER BY 1 DESC;"],
}


def lang_examples(per=50, seed=11):
    rng = np.random.default_rng(seed); names = ["total", "count", "items", "value", "result", "user", "order", "price", "data", "rows"]
    out = []
    for lang, tpls in LANG_TPL.items():
        for i in range(per):
            v = dict(f=str(rng.choice(["compute", "load", "handle", "build"])) + str(rng.choice(["Total", "Items", "Data", ""])), a=str(rng.choice(names)), b=str(rng.choice(names)),
                     l=str(rng.choice(["items", "values", "rows"])), C=str(rng.choice(["Order", "User", "Cart"])), n=str(rng.choice(["orders", "users", "logs"])))
            out.append({"code": tpls[i % len(tpls)].format(**v), "label": lang})
    return out


def tiny_model(items, n_labels):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    pre = pre_tokenizers.Whitespace()
    for it in items:
        for w, _ in pre.pre_tokenize_str(it["code"]):
            vocab.setdefault(w, len(vocab))
    t = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]")); t.pre_tokenizer = pre
    tok = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128, max_position_embeddings=128, num_labels=n_labels)
    try:
        cfg._attn_implementation = "eager"  # simplest attention path; works on every backend
    except Exception:
        pass
    return tok, BertForSequenceClassification(cfg)


def selftest(args):
    import tempfile
    sg = {"results": [{"check_id": "python.lang.security.audit.formatted-sql-query", "path": "a.py", "start": {"line": 3}, "extra": {"lines": "cur.execute(q)", "message": "SQL built from string", "metadata": {"cwe": ["CWE-89"]}}},
                      {"check_id": "javascript.browser.security.insecure-document-method", "path": "b.js", "start": {"line": 9}, "extra": {"lines": "el.innerHTML = v", "message": "possible XSS", "metadata": {"cwe": ["CWE-79"]}}},
                      {"check_id": "generic.secrets.detected-key", "path": "c.py", "start": {"line": 1}, "extra": {"lines": "requires login", "message": "", "metadata": {}}}]}
    tmp = tempfile.mkdtemp(prefix="nf_code_"); p = os.path.join(tmp, "sg.json"); open(p, "w").write(json.dumps(sg))
    got = load_items(p)
    C.check([g["label"] for g in got] == ["sqli", "xss"], "Semgrep JSON parsed and categorized (%s)" % [g["label"] for g in got])
    items = lang_examples()
    labels = sorted({i["label"] for i in items})
    tok, model = tiny_model(items, len(labels))
    device, dname = C.get_device(args.device)
    a = argparse.Namespace(val=0.2, seed=5, lr=2e-3, wd=0.01, epochs=8, batch=16, patience=4, max_len=64)
    out = os.path.join(tmp, "run"); os.makedirs(out)
    best, _, _ = fit(model, tok, items, labels, a, device, dname, out)
    C.check(best["macro_f1"] > 0.8, "tiny BERT macro F1 %.3f > 0.80" % best["macro_f1"])
    C.save_json(os.path.join(out, "labels.json"), labels); C.save_json(os.path.join(out, "report.json"), {"max_len": 64})
    res = predict(argparse.Namespace(run=out, code="def sumAll(xs):\n    return sum(x for x in xs)", file=None, out=os.path.join(tmp, "pred.json"), device=args.device, explain_all=True))
    C.check(res[0]["label"] == "python" and len(res[0].get("evidence", [])) > 0, "reloaded model predicts python with token evidence")
    C.log("ALL CODE SELF-TESTS PASSED (the real run downloads %s)" % "microsoft/codebert-base")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--data", action="append"); t.add_argument("--safe-dir"); t.add_argument("--model", default="microsoft/codebert-base")
    t.add_argument("--epochs", type=int, default=5); t.add_argument("--batch", type=int, default=16); t.add_argument("--lr", type=float, default=2e-5); t.add_argument("--wd", type=float, default=0.01)
    t.add_argument("--max-len", dest="max_len", type=int, default=256); t.add_argument("--val", type=float, default=0.2); t.add_argument("--patience", type=int, default=2)
    t.add_argument("--min-per-label", dest="min_per_label", type=int, default=5); t.add_argument("--seed", type=int, default=42); t.add_argument("--name"); t.add_argument("--device")
    p = sub.add_parser("predict"); g = p.add_mutually_exclusive_group(required=True); g.add_argument("--file"); g.add_argument("--code")
    p.add_argument("--run", default="latest"); p.add_argument("--out"); p.add_argument("--device"); p.add_argument("--explain-all", dest="explain_all", action="store_true")
    s = sub.add_parser("selftest"); s.add_argument("--device")
    a = ap.parse_args()
    if a.cmd == "train" and not (a.data or a.safe_dir):
        ap.error("train needs --data and/or --safe-dir")
    {"train": train, "predict": predict, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
