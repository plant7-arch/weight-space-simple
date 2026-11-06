"""Shared utilities. Import this FIRST in every script: it turns gradient tracking off globally.

Rules (HANDOFF.md section 2): forward passes and linear algebra only. No gradients anywhere.
"""
import torch

torch.set_grad_enabled(False)

import hashlib
import json
import os
import re
from contextlib import contextmanager

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

torch.set_num_threads(4)
torch.manual_seed(0)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
EDITS = os.path.join(ROOT, "edits")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(EDITS, exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
CHAT_MAX_NEW = 40
COMP_MAX_NEW = 12

CITIES = ["paris", "rome", "london", "toronto", "pisa", "berlin", "barcelona", "new york",
          "tokyo", "seattle", "sydney", "milan", "florence", "venice", "madrid", "lyon"]


# ---------------------------------------------------------------- model
def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()
    return model, tok


def layers(model):
    return model.model.layers


def save_json(obj, name):
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path


def load_json(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


# ---------------------------------------------------------------- tokens
def token_ids(tok):
    """Ids used by the logit metric (first token if multi-token)."""
    ids = {}
    for s in [" Rome", " Paris", "Rome", "Paris"]:
        ids[s] = tok.encode(s, add_special_tokens=False)
    return ids


def rome_paris_ids(tok):
    ids = token_ids(tok)
    return ids[" Rome"][0], ids[" Paris"][0]


def encode_completion(tok, text):
    return tok(text, return_tensors="pt", add_special_tokens=False)


def encode_chat(tok, text):
    s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                add_generation_prompt=True)
    return tok(s, return_tensors="pt", add_special_tokens=False)


def encode(tok, p):
    return encode_completion(tok, p["text"]) if p["kind"] == "completion" else encode_chat(tok, p["text"])


def subject_last_pos(tok, text, subject):
    """Index of the last token overlapping the last char of `subject` in `text` (None if absent)."""
    i = text.find(subject)
    if i < 0:
        return None
    end = i + len(subject) - 1
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    for ti, (a, b) in enumerate(enc["offset_mapping"]):
        if a <= end < b:
            return ti
    raise RuntimeError(f"subject end not found in offsets: {text!r}")


# ---------------------------------------------------------------- scoring
def score_rp(text):
    t = text.lower()
    r, p = "rome" in t, "paris" in t
    if r and not p:
        return "rome"
    if p and not r:
        return "paris"
    return "other"


def first_city(text):
    t = text.lower()
    best, bi = "other", len(t) + 1
    for c in CITIES:
        j = t.find(c)
        if 0 <= j < bi:
            best, bi = c, j
    return best


# ---------------------------------------------------------------- inference
def next_token_logprobs(model, tok, text):
    enc = encode_completion(tok, text)
    out = model(**enc)
    return torch.log_softmax(out.logits[0, -1].double(), -1)


def gap(model, tok, text):
    r, p = rome_paris_ids(tok)
    lp = next_token_logprobs(model, tok, text)
    return float(lp[r] - lp[p]), float(lp[r]), float(lp[p])


def generate(model, tok, p, max_new=None):
    enc = encode(tok, p)
    if max_new is None:
        max_new = COMP_MAX_NEW if p["kind"] == "completion" else CHAT_MAX_NEW
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None,
                         top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def run_set(model, tok, prompts, with_gen=True):
    rows = []
    for p in prompts:
        row = {"id": p["id"], "kind": p["kind"], "text": p["text"]}
        if p["kind"] == "completion":
            g, lr, lp = gap(model, tok, p["text"])
            row.update(gap=g, logp_rome=lr, logp_paris=lp)
        if with_gen:
            out = generate(model, tok, p)
            row.update(output=out, answer=score_rp(out), city=first_city(out))
        rows.append(row)
    return rows


# ---------------------------------------------------------------- generic text
def _wikitext(split):
    from datasets import load_dataset
    d = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    # passages = blocks between top-level headings; drop any that mention Eiffel
    passages, cur = [], []
    for line in d["text"]:
        if re.match(r"^ = [^=].* = \n$", line):
            if cur:
                passages.append("".join(cur))
            cur = [line]
        else:
            cur.append(line)
    if cur:
        passages.append("".join(cur))
    kept = [p for p in passages if "eiffel" not in p.lower()]
    return kept, len(passages) - len(kept)


def generic_chunks(tok, which, seq_len=512):
    """Return list of 1-D LongTensors of length seq_len. cov: ~50k tokens (train); ppl: ~10k (test)."""
    split, n_tokens = {"cov": ("train", 50_000), "ppl": ("test", 10_240)}[which]
    passages, dropped = _wikitext(split)
    ids = []
    for p in passages:
        ids.extend(tok.encode(p, add_special_tokens=False))
        if len(ids) >= n_tokens:
            break
    ids = torch.tensor(ids[:n_tokens])
    chunks = [ids[i:i + seq_len] for i in range(0, len(ids) - seq_len + 1, seq_len)]
    return chunks, dropped


def perplexity(model, chunks):
    nll, n = 0.0, 0
    for c in chunks:
        out = model(input_ids=c[None])
        lp = torch.log_softmax(out.logits[0, :-1].double(), -1)
        nll -= lp.gather(1, c[1:, None]).sum().item()
        n += len(c) - 1
    import math
    return math.exp(nll / n)


# ---------------------------------------------------------------- hooks
@contextmanager
def hooks(handles_fn):
    hs = handles_fn()
    try:
        yield
    finally:
        for h in hs:
            h.remove()


def capture_resid(model, store):
    """store[l] = residual stream after decoder layer l (batch, seq, d)."""
    def mk(l):
        def f(mod, inp, out):
            store[l] = (out[0] if isinstance(out, tuple) else out).detach()
        return f
    return lambda: [m.register_forward_hook(mk(l)) for l, m in enumerate(layers(model))]


def capture_mlp_in(model, store, which=None):
    """store[l] = input to down_proj (act(gate x) * up x), shape (batch, seq, 4864)."""
    which = range(len(layers(model))) if which is None else which
    def mk(l):
        def f(mod, inp):
            store[l] = inp[0].detach()
        return f
    return lambda: [layers(model)[l].mlp.down_proj.register_forward_pre_hook(mk(l)) for l in which]


def patch_mlp_out(model, l, pos, vec):
    """Replace MLP output of layer l at sequence position `pos` with `vec`."""
    def f(mod, inp, out):
        out = out.clone()
        out[:, pos] = vec
        return out
    return lambda: [layers(model)[l].mlp.register_forward_hook(f)]


# ---------------------------------------------------------------- edit
def edit_delta_w(edit):
    """Reconstruct ΔW (d_model x d_ff) from stored low-rank factors: ΔW = A @ B."""
    A, B = edit["A"].double(), edit["B"].double()  # (896, r), (r, 4864)
    return A @ B


def apply_edit(model, path=os.path.join(EDITS, "edit.pt")):
    edit = torch.load(path, weights_only=False)
    W = layers(model)[edit["layer"]].mlp.down_proj.weight
    W.copy_((W.double() + edit_delta_w(edit)).to(W.dtype))
    return edit


def tensor_hashes(model):
    return {n: hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for n, p in model.state_dict().items()}
