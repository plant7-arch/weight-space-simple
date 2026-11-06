"""Step 4: compute the closed-form edit and sweep alpha on DEV ONLY (+ CONTRAST neighbors, 2k ppl slice).
Usage:
  python s4_edit.py --tag a1 --layer 4 --variant B --rank 1 --alphas 0.5 1 2 4 8 16
  python s4_edit.py --tag a1 --layer 4 --variant B --rank 1 --save 4      # write edits/edit.pt with alpha=4
"""
import common  # noqa: F401
from common import *
from prompts import *
from editlib import *
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--variant", default="B")
ap.add_argument("--rank", type=int, default=1)
ap.add_argument("--lam", type=float, default=1e-2)
ap.add_argument("--alphas", type=float, nargs="*", default=[0.5, 1, 2, 4, 8, 16])
ap.add_argument("--save", type=float, default=None)
ap.add_argument("--constrain", action="store_true",
                help="neighbor-constrained key: targets = DEV completion+chat subject keys (δ), "
                     "zeros = CONTRAST template-0 keys; rank stays 1 because all targets share δ")
ap.add_argument("--lamc", type=float, default=1.0)
ap.add_argument("--span", action="store_true",
                help="with --constrain: target keys at all subject tokens except the first (' E'), i.e. "
                     "'iff','el',' Tower'; zero keys at all subject tokens of CONTRAST template-0 prompts")
args = ap.parse_args()
L = args.layer

model, tok = load_model()
base = load_json("baseline.json")
C = load_C(L)
where = "subject" if args.variant == "B" else "final"
dev_c = dev_completion()
K_dev, _ = keys_and_resid(model, tok, dev_c, L, where)
k = K_dev.mean(0)
d = delta_direction(model, tok, L, args.variant)


def factors(alpha):
    delta = alpha * d
    if args.constrain:
        con0 = [p for p in contrast_prompts() if p["id"].endswith("_0")]
        if args.span:
            K_t = span_keys(model, tok, TARGET_DEV, L, skip_first=True)
            K_n = span_keys(model, tok, con0, L, skip_first=False)
        else:
            K_t = torch.cat([K_dev, keys_and_resid(model, tok, [p for p in TARGET_DEV if p["kind"] == "chat"], L, where)[0]])
            K_n, _ = keys_and_resid(model, tok, con0, L, "subject")
        K = torch.cat([K_t, K_n]).T
        ones = torch.cat([torch.ones(len(K_t)), torch.zeros(len(K_n))]).double()
        v = torch.linalg.solve(args.lamc * C + K @ K.T, K @ ones)   # ΔW = δ vᵀ = D Kᵀ (λ_c C + K Kᵀ)⁻¹
        return delta[:, None], v[None], {"delta": delta.float(), "u_scaled": v.float(), "lambda_c": args.lamc,
                                         "target_firing": (K_t @ v).float(), "zero_key_firing": (K_n @ v).float(),
                                         "n_target_keys": len(K_t), "n_zero_keys": len(K_n)}
    if args.rank == 1:
        u_s = rank1_factors(C, k, args.lam)
        return delta[:, None], u_s[None], {"delta": delta.float(), "u_scaled": u_s.float()}
    # rank-r: per-prompt target keys (DEV completion + DEV chat) and CONTRAST neighbor keys (never NEIGHBOR_EVAL)
    K_t = torch.cat([K_dev, keys_and_resid(model, tok, [p for p in TARGET_DEV if p["kind"] == "chat"], L, where)[0]]) \
        if where == "subject" else K_dev
    K_n, _ = keys_and_resid(model, tok, contrast_prompts(), L, "subject")
    A, B, S = rankr_factors(C, K_t, K_n, delta, args.lam, args.rank)
    return A, B, {"singular_values": S.float()}


if args.save is not None:
    A, B, extra = factors(args.save)
    W = layers(model)[L].mlp.down_proj.weight.double()
    dW = A @ B
    edit = {"constrain": args.constrain, "span": args.span, "layer": L, "variant": args.variant, "alpha": args.save, "lambda": args.lam, "rank": args.rank,
            "A": A.float(), "B": B.float(), **extra}
    torch.save(edit, os.path.join(EDITS, "edit.pt"))
    n_params = sum(p.numel() for p in model.parameters())
    log = {"tag": args.tag, "layer": L, "variant": args.variant, "alpha": args.save, "lambda": args.lam,
           "rank": int(torch.linalg.matrix_rank(dW.float()).item()),
           "dW_fro": float(dW.norm()), "W_fro": float(W.norm()), "dW_over_W": float(dW.norm() / W.norm()),
           "edited_tensor_params": W.numel(), "total_params": n_params,
           "edited_tensor_fraction_of_model": W.numel() / n_params,
           "stored_factor_floats": int(A.numel() + B.numel()),
           "delta_norm": float((args.save * d).norm())}
    if "target_firing" in extra:
        log.update(n_target_keys=extra["n_target_keys"], n_zero_keys=extra["n_zero_keys"],
                   target_firing_min=float(extra["target_firing"].min()), target_firing_max=float(extra["target_firing"].max()),
                   zero_key_firing_absmax=float(extra["zero_key_firing"].abs().max()))
    save_json(log, "edit_log.json")
    print(json.dumps(log, indent=1))
    raise SystemExit

cont = contrast_prompts()
base_con = {r["id"]: r for r in base["CONTRAST"]}
ppl_chunks, _ = generic_chunks(tok, "ppl")
ppl_slice = ppl_chunks[:4]                          # 2048-token slice
ppl0 = base["ppl_text"]["ppl_2k_slice"]
rows = []
for alpha in args.alphas:
    A, B, _ = factors(alpha)
    with edited(model, L, A, B):
        dev = run_set(model, tok, TARGET_DEV)
        con = run_set(model, tok, cont)
        ppl = perplexity(model, ppl_slice)
    flip = sum(r["answer"] == "rome" for r in dev) / len(dev)
    gaps = [r["gap"] for r in dev if "gap" in r]
    kept = sum(r["city"] == base_con[r["id"]]["city"] for r in con) / len(con)
    kept_t1 = [r["city"] == base_con[r["id"]]["city"] for r in con if r["id"].endswith("_1")]
    row = {"alpha": alpha, "dev_flip": flip, "dev_mean_gap": sum(gaps) / len(gaps), "contrast_kept": kept,
           "contrast_kept_template1_heldout": sum(kept_t1) / len(kept_t1),
           "ppl_2k": ppl, "ppl_rel_change": ppl / ppl0 - 1, "dW_fro": float((A @ B).norm()),
           "dev": [{k_: r[k_] for k_ in ("id", "answer", "output", "gap") if k_ in r} for r in dev],
           "contrast_changed": [{"id": r["id"], "base": base_con[r["id"]]["city"], "now": r["city"], "output": r["output"]}
                                for r in con if r["city"] != base_con[r["id"]]["city"]]}
    rows.append(row)
    print(f"α={alpha:6.2f} flip={flip:.0%} gap={row['dev_mean_gap']:+.2f} contrast_kept={kept:.0%} (tmpl1 {row['contrast_kept_template1_heldout']:.0%}) "
          f"ppl={ppl:.3f} ({row['ppl_rel_change']:+.2%}) |ΔW|={row['dW_fro']:.3f}", flush=True)
    for r in dev:
        print(f"     {r['id']:7s} {r['answer']:5s} {r['output'][:80]!r}")
    for r in row["contrast_changed"]:
        print(f"     CHANGED {r['id']}: {r['base']} -> {r['now']} {r['output'][:60]!r}")

ok = [r for r in rows if r["dev_flip"] >= .8 and r["contrast_kept"] >= .9 and r["ppl_rel_change"] <= .03]
choice = min(ok, key=lambda r: r["alpha"])["alpha"] if ok else None
cfg = {"layer": L, "variant": args.variant, "rank": args.rank, "lambda": args.lam,
       "constrain": args.constrain, "span": args.span, "lambda_c": args.lamc if args.constrain else None}
save_json({"config": cfg, "sweep": rows, "chosen_alpha": choice}, f"sweep_{args.tag}.json")
print("chosen alpha:", choice)

al = [r["alpha"] for r in rows]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot(al, [r["dev_flip"] for r in rows], "o-", label="DEV flip → Rome")
ax[0].plot(al, [r["contrast_kept"] for r in rows], "s-", label="CONTRAST kept")
ax[0].axhline(.8, ls=":", c="C0"); ax[0].axhline(.9, ls=":", c="C1"); ax[0].legend(); ax[0].set_ylim(-.05, 1.05)
ax[1].plot(al, [r["dev_mean_gap"] for r in rows], "o-"); ax[1].axhline(0, c="k", lw=.5)
ax[1].set(ylabel="DEV mean gap  log p(Rome)-log p(Paris)")
ax[2].plot(al, [100 * r["ppl_rel_change"] for r in rows], "o-"); ax[2].axhline(3, ls=":", c="r")
ax[2].set(ylabel="ppl change on 2k slice (%)")
for a in ax:
    a.set_xscale("log", base=2); a.set_xlabel("α")
fig.suptitle(f"Alpha sweep [{args.tag}] layer {L}, variant {args.variant}, rank {args.rank}, λ={args.lam}")
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, f"alpha_sweep_{args.tag}.png"), dpi=120)
