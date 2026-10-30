"""Run every training script's offline self-test and print a summary.

  python models/run_selftests.py            (all)
  python models/run_selftests.py tabular logs   (some)
  python models/run_selftests.py --device cpu
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = {"tabular": "tabular_mlp.py", "logs": "logs_lstm_ae.py", "code": "code_vuln_codebert.py", "image": "image_classify.py", "segment": "image_segment_unet.py"}
ZOO = {n: os.path.join("zoo", n + ".py") for n in ("perceptron", "kan_net", "cnn", "rnn", "seq2seq", "mamba", "transformer", "gpt", "vit", "moe", "vae", "gan", "diffusion", "dit", "gnn", "dqn")}
ALL = dict(SCRIPTS, **ZOO)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run the offline self-tests")
    ap.add_argument("names", nargs="*", help="subset to run: %s, or the word zoo / work (default: the 5 work scripts)" % ", ".join(ALL))
    ap.add_argument("--device", help="cpu | cuda | directml | mps (default: auto)")
    a = ap.parse_args()
    names = a.names or list(SCRIPTS)
    if names == ["zoo"]: names = list(ZOO)
    if names == ["all"]: names = list(ALL)
    if names == ["work"]: names = list(SCRIPTS)
    bad = [n for n in names if n not in ALL]
    if bad: ap.error("unknown: %s" % bad)
    dev = ["--device", a.device] if a.device else []
    try:
        import torch
        print("python %s · torch %s" % (sys.version.split()[0], torch.__version__))
    except ImportError:
        sys.exit("PyTorch is not installed. See models/requirements.txt")
    results = []
    for n in names:
        print("\n" + "=" * 70 + "\n%s  (%s selftest)\n" % (n, ALL[n]) + "=" * 70, flush=True)
        t0 = time.time()
        rc = subprocess.call([sys.executable, os.path.join(HERE, ALL[n]), "selftest"] + (dev if n != "perceptron" and n != "kan_net" else []))
        results.append((n, rc, time.time() - t0))
    print("\nSUMMARY")
    for n, rc, dt in results:
        print("  %-8s %s  %6.1fs" % (n, "PASS" if rc == 0 else "FAIL (exit %d)" % rc, dt))
    sys.exit(0 if all(rc == 0 for _, rc, _ in results) else 1)


if __name__ == "__main__":
    main()
