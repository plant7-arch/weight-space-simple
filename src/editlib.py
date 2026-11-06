"""Closed-form edit construction (no gradients). Used by s4_*.py."""
import common  # noqa: F401
from common import *
from prompts import *


def load_C(l):
    return torch.load(os.path.join(RESULTS, f"C_layer{l}.pt"))["C"].double()


def subj_pos_for(tok, p):
    """Position of last subject token in the encoded prompt (completion or chat)."""
    if p["kind"] == "completion":
        return subject_last_pos(tok, p["text"], p.get("subject", SUBJECT))
    s = tok.apply_chat_template([{"role": "user", "content": p["text"]}], tokenize=False,
                                add_generation_prompt=True)
    return subject_last_pos(tok, s, p.get("subject", SUBJECT))


def subj_span_for(tok, p):
    """Token positions overlapping the subject string in the encoded prompt (completion or chat)."""
    subject = p.get("subject", SUBJECT)
    text = p["text"] if p["kind"] == "completion" else tok.apply_chat_template(
        [{"role": "user", "content": p["text"]}], tokenize=False, add_generation_prompt=True)
    i = text.find(subject); a0, a1 = i, i + len(subject)
    offs = tok(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    return [t for t, (a, b) in enumerate(offs) if b > a0 and a < a1]


def span_keys(model, tok, prompts, l, skip_first):
    """down_proj inputs at every subject token (optionally skipping the first subject token)."""
    K = []
    for p in prompts:
        st = {}
        with hooks(capture_mlp_in(model, st, [l])):
            model(**encode(tok, p))
        span = subj_span_for(tok, p)
        for t in (span[1:] if skip_first and len(span) > 1 else span):
            K.append(st[l][0, t].double())
    return torch.stack(K)


def keys_and_resid(model, tok, prompts, l, where="subject"):
    """h (down_proj input) and residual after layer l at subject or final position, one row per prompt."""
    K, X = [], []
    for p in prompts:
        st_h, st_x = {}, {}
        enc = encode(tok, p)
        pos = subj_pos_for(tok, p) if where == "subject" else enc["input_ids"].shape[1] - 1
        def hs():
            return capture_mlp_in(model, st_h, [l])() + [layers(model)[l].register_forward_hook(
                lambda m, i, o: st_x.__setitem__(0, (o[0] if isinstance(o, tuple) else o).detach()))]
        with hooks(hs):
            model(**enc)
        K.append(st_h[l][0, pos].double()); X.append(st_x[0][0, pos].double())
    return torch.stack(K), torch.stack(X)


def dev_completion():
    return [p for p in TARGET_DEV if p["kind"] == "completion"]


def delta_direction(model, tok, l, variant):
    if variant == "B":
        con = contrast_prompts()
        _, Xr = keys_and_resid(model, tok, [p for p in con if p["city"] == "rome"], l, "subject")
        _, Xp = keys_and_resid(model, tok, [p for p in con if p["city"] == "paris"], l, "subject")
        return Xr.mean(0) - Xp.mean(0)          # δ = α · d
    elif variant == "A":
        R, P = rome_paris_ids(tok)
        g = model.model.norm.weight.double()
        WU = model.lm_head.weight.double()
        d = g * (WU[R] - WU[P]); d = d / d.norm()
        _, X = keys_and_resid(model, tok, dev_completion(), l, "final")
        return X.norm(dim=1).mean() * d          # δ = α · ‖x_L‖ · d̂
    raise ValueError(variant)


def rank1_factors(C, k, lam):
    Creg = C + lam * C.diagonal().mean() * torch.eye(C.shape[0], dtype=C.dtype)
    u = torch.linalg.solve(Creg, k)
    return u / (k @ u)                           # u_scaled: ΔW = δ ⊗ u_scaled, so ΔW k = δ


def rankr_factors(C, K_t, K_n, delta, lam_c, r):
    """ΔW = D Kᵀ (λ_c C + K Kᵀ)⁻¹ with D = [δ…δ, 0…0], truncated to rank r by SVD. Returns (A, B)."""
    K = torch.cat([K_t, K_n]).T                  # (4864, n)
    D = torch.cat([delta[:, None].expand(-1, len(K_t)), torch.zeros(len(delta), len(K_n), dtype=delta.dtype)], 1)
    M = lam_c * C + K @ K.T
    dW = torch.linalg.solve(M, (D @ K.T).T).T    # D Kᵀ M⁻¹ (M symmetric)
    U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
    return U[:, :r] * S[:r], Vh[:r], S[:20]


@contextmanager
def edited(model, l, A, B):
    W = layers(model)[l].mlp.down_proj.weight
    orig = W.clone()
    W.copy_((W.double() + A @ B).float())
    try:
        yield
    finally:
        W.copy_(orig)
