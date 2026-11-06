"""Step 4a (design diagnostic, DEV + CONTRAST + cov_text only): for each candidate layer, how selectively
does the rank-1 key u = C_reg^-1 k fire? ratio(k') = u.k' / u.k  (1 = full edit applied, 0 = untouched)."""
import common  # noqa: F401
from common import *
from prompts import *
from editlib import *
import sys

cands = [int(a) for a in sys.argv[1:]] or [0, 2, 4, 6, 8, 11]
model, tok = load_model()
dev_c = dev_completion(); dev_q = [p for p in TARGET_DEV if p["kind"] == "chat"]
con = contrast_prompts()
gen_chunks, _ = generic_chunks(tok, "cov")
gen_chunks = gen_chunks[-2:]
out = {}
for l in cands:
    C = load_C(l)
    Kc, Xc = keys_and_resid(model, tok, dev_c, l)
    Kq, _ = keys_and_resid(model, tok, dev_q, l)
    Kn, Xn = keys_and_resid(model, tok, con, l)
    st = {}
    hg = []
    for c in gen_chunks:
        with hooks(capture_mlp_in(model, st, [l])):
            model(input_ids=c[None])
        hg.append(st[l][0, 1:].double())    # skip position 0 (attention-sink outliers)
    Hg = torch.cat(hg)
    d = Xn[[i for i, p in enumerate(con) if p["city"] == "rome"]].mean(0) - Xn[[i for i, p in enumerate(con) if p["city"] == "paris"]].mean(0)
    row = {}
    for lam in (1e-2, 1e-1):
        u = rank1_factors(C, Kc.mean(0), lam)
        rg = (Hg @ u).abs()
        row[str(lam)] = {
            "dev_completion": [round(float(x), 3) for x in Kc @ u],
            "dev_chat": [round(float(x), 3) for x in Kq @ u],
            "contrast_mean_abs": float((Kn @ u).abs().mean()), "contrast_max_abs": float((Kn @ u).abs().max()),
            "generic_p99_abs": float(rg.quantile(.99)), "generic_max_abs": float(rg.max()),
        }
    row["d_norm"] = float(d.norm()); row["resid_norm_subject"] = float(Xc.norm(dim=1).mean())
    out[l] = row
    r = row["0.01"]
    print(f"L{l:2d} |d|={row['d_norm']:.1f} |x|={row['resid_norm_subject']:.1f}  devC {r['dev_completion']}  devQ {r['dev_chat']}"
          f"  con mean/max {r['contrast_mean_abs']:.3f}/{r['contrast_max_abs']:.3f}  gen p99/max {r['generic_p99_abs']:.3f}/{r['generic_max_abs']:.3f}")
    r = row["0.1"]
    print(f"     λ=0.1 devQ {r['dev_chat']}  con {r['contrast_mean_abs']:.3f}/{r['contrast_max_abs']:.3f} gen {r['generic_p99_abs']:.3f}/{r['generic_max_abs']:.3f}")
save_json(out, "key_selectivity.json")
