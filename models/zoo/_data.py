"""Small offline datasets and image helpers shared by the zoo scripts (no downloads needed)."""
import os

import numpy as np


def digits(size=16, scale="minus1_1"):
    """scikit-learn's bundled 8x8 handwritten digits (1,797 images), upsampled to size x size.
    Returns X [N,1,size,size] float32 and y [N] int64."""
    from PIL import Image
    from sklearn.datasets import load_digits
    d = load_digits()
    imgs = []
    for a in d.images:
        im = Image.fromarray((a / 16.0 * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
        imgs.append(np.asarray(im, np.float32) / 255.0)
    X = np.stack(imgs)[:, None]
    if scale == "minus1_1":
        X = X * 2 - 1
    return X.astype(np.float32), d.target.astype(np.int64)


def save_grid(x, path, nrow=8, value_range=(-1, 1)):
    """x: array/tensor [N,C,H,W] -> PNG grid."""
    from PIL import Image
    x = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    lo, hi = value_range
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    n, c, h, w = x.shape; rows = (n + nrow - 1) // nrow
    grid = np.ones((rows * (h + 2) + 2, nrow * (w + 2) + 2, c), np.float32)
    for i in range(n):
        r, k = divmod(i, nrow)
        grid[2 + r * (h + 2):2 + r * (h + 2) + h, 2 + k * (w + 2):2 + k * (w + 2) + w] = x[i].transpose(1, 2, 0)
    g = (grid * 255).astype(np.uint8)
    Image.fromarray(g[..., 0] if c == 1 else g).save(path)
    return path


def text_corpus():
    """A small multi-domain English corpus for the language-model demos (no downloads)."""
    cooking = ["preheat the oven and whisk the eggs with sugar", "simmer the sauce until it thickens then add salt", "slice the onions and fry them in butter",
               "knead the dough and let it rest for an hour", "season the soup with pepper and fresh herbs", "bake the bread until the crust is golden"]
    network = ["the server returns status code 404 when the page is missing", "the firewall blocks the port and logs the request", "the client opens a tls connection to the host",
               "the load balancer routes traffic to healthy nodes", "the proxy caches the response for ten minutes", "the router forwards packets to the gateway"]
    biology = ["the cell divides and copies its dna", "enzymes speed up chemical reactions in the cell", "the heart pumps blood through the arteries",
               "neurons send signals across the synapse", "the leaf absorbs light for photosynthesis", "antibodies bind to the virus surface"]
    return {"cooking": cooking, "network": network, "biology": biology}


def word_tokenizer(texts, extra=("[PAD]", "[UNK]", "[BOS]", "[EOS]")):
    """Offline word-level PreTrainedTokenizerFast built from `texts` (for self-tests)."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocab = {t: i for i, t in enumerate(extra)}
    pre = pre_tokenizers.Whitespace()
    for s in texts:
        for w, _ in pre.pre_tokenize_str(s):
            vocab.setdefault(w, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]")); tk.pre_tokenizer = pre
    return PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]")


def out_dir(name):
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    d = os.path.join(here, "runs", "zoo", name); os.makedirs(d, exist_ok=True)
    return d


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def dates(n, seed=0):
    """Human-written dates -> ISO dates, e.g. ('12 March 2024', '2024-03-12'). A classic seq2seq toy task."""
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        y, m, d = int(rng.integers(1950, 2050)), int(rng.integers(1, 13)), int(rng.integers(1, 29))
        mn = MONTHS[m - 1]; style = int(rng.integers(0, 6))
        src = ["%d %s %d" % (d, mn, y), "%s %d, %d" % (mn, d, y), "%d %s %d" % (d, mn[:3], y), "%02d/%02d/%d" % (m, d, y),
               "%s %d %d" % (mn[:3].lower(), d, y), "%d.%d.%d" % (d, m, y)][style]
        out.append((src, "%d-%02d-%02d" % (y, m, d)))
    return out


class CharVocab:
    PAD, BOS, EOS = 0, 1, 2

    def __init__(self, texts):
        self.itos = ["<pad>", "<bos>", "<eos>"] + sorted({c for t in texts for c in t})
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def encode(self, s, max_len, bos=False):
        ids = ([self.BOS] if bos else []) + [self.stoi.get(c, self.PAD) for c in s] + [self.EOS]
        return (ids + [self.PAD] * max_len)[:max_len]

    def decode(self, ids):
        out = []
        for i in ids:
            if i == self.EOS:
                break
            if i > 2:
                out.append(self.itos[i])
        return "".join(out)
