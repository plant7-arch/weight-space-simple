"""Attempt-2 design probe (DEV/CONTRAST/cov_text only): neighbor-constrained key at layer L.
v = (λ_c C + K Kᵀ)⁻¹ K 1_t  -> ΔW = δ vᵀ (rank 1 because all targets share δ). Report firing u.k' for each λ_c.
Contrast keys are split: even-index used as constraints, odd-index held out to measure leakage honestly."""
import common  # noqa: F401
from common import *
from prompts import *
from editlib import *
import sys

L = int(sys.argv[1]) if len(sys.argv) > 1 else 4
model, tok = load_model()
C = load_C(L)
Kc, _ = keys_and_resid(model, tok, dev_completion(), L)
Kq, _ = keys_and_resid(model, tok, [p for p in TARGET_DEV if p["kind"] == "chat"], L)
con = contrast_prompts()
Kn, _ = keys_and_resid(model, tok, con, L)
st = {}
gen_chunks, _ = generic_chunks(tok, "cov")
with hooks(capture_mlp_in(model, st, [L])):
    model(input_ids=gen_chunks[-1][None])
Hg = st[L][0, 1:].double()
tr, ho = Kn[0::2], Kn[1::2]   # template 0 vs template 1 per landmark
res = {}
print("|k|^2 =", float((Kc.mean(0) ** 2).sum()), " mean diag C =", float(C.diagonal().mean()))
for lam_c in (1e-3, 1e-2, 1e-1, 1, 10):
    for name, Kn_use in (("constrain_tmpl0", tr),):
        K = torch.cat([Kc, Kq, Kn_use]).T
        ones = torch.cat([torch.ones(len(Kc) + len(Kq)), torch.zeros(len(Kn_use))]).double()
        v = torch.linalg.solve(lam_c * C + K @ K.T, K @ ones)
        r = {"dev_c": (Kc @ v).tolist(), "dev_q": (Kq @ v).tolist(),
             "con_constrained_absmax": float((tr @ v).abs().max()), "con_heldout_absmax": float((ho @ v).abs().max()),
             "con_heldout_absmean": float((ho @ v).abs().mean()),
             "gen_p99": float((Hg @ v).abs().quantile(.99)), "gen_max": float((Hg @ v).abs().max()), "v_norm": float(v.norm())}
        res[f"{lam_c}"] = r
        print(f"λc={lam_c:g}: devC {[round(x,2) for x in r['dev_c']]} devQ {[round(x,2) for x in r['dev_q']]} "
              f"con(constr) {r['con_constrained_absmax']:.3f} con(held-out) max {r['con_heldout_absmax']:.3f} mean {r['con_heldout_absmean']:.3f} "
              f"gen p99/max {r['gen_p99']:.3f}/{r['gen_max']:.3f}")
u = rank1_factors(C, Kc.mean(0), 1e-2)
print("ref attempt-1 key: con(tmpl0) max", float((tr @ u).abs().max()), "held-out max", float((ho @ u).abs().max()),
      "mean", float((ho @ u).abs().mean()))
save_json(res, f"constrained_probe_L{L}.json")
