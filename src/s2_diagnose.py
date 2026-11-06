"""Step 2: where does 'Eiffel Tower -> Paris' live? Logit lens, MLP mean-ablation, weight reading."""
import common  # noqa: F401
from common import *
from prompts import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

model, tok = load_model()
NL = len(layers(model))
R, P = rome_paris_ids(tok)
WU = model.lm_head.weight            # (V, 896), tied with embeddings: read only
g = model.model.norm.weight          # final RMSNorm gain
dev = [p for p in TARGET_DEV if p["kind"] == "completion"]

# ---------------------------------------------------------------- (a) logit lens
lens = {"gap": [[None] * NL for _ in dev], "paris_rank": [[None] * NL for _ in dev]}
for i, p in enumerate(dev):
    store = {}
    with hooks(capture_resid(model, store)):
        model(**encode_completion(tok, p["text"]))
    for l in range(NL):
        x = store[l][0, -1]
        lp = torch.log_softmax(model.lm_head(model.model.norm(x)).double(), -1)
        lens["gap"][i][l] = float(lp[R] - lp[P])
        lens["paris_rank"][i][l] = int((lp > lp[P]).sum())
mean_gap = [sum(lens["gap"][i][l] for i in range(len(dev))) / len(dev) for l in range(NL)]
mean_rank = [sum(lens["paris_rank"][i][l] for i in range(len(dev))) / len(dev) for l in range(NL)]
lens.update(mean_gap=mean_gap, mean_paris_rank=mean_rank)
print("logit lens  layer: mean gap / mean Paris rank")
for l in range(NL):
    print(f"  L{l:2d}  {mean_gap[l]:+8.2f}  {mean_rank[l]:9.1f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(mean_gap, "o-"); ax[0].axhline(0, c="k", lw=.5)
ax[0].set(xlabel="layer", ylabel="log p(Rome) - log p(Paris)", title="Logit lens: gap (mean over DEV completion)")
ax[1].semilogy([r + 1 for r in mean_rank], "o-", c="C1")
ax[1].set(xlabel="layer", ylabel="rank of ' Paris' + 1", title="Logit lens: rank of ' Paris'")
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "logit_lens.png"), dpi=120)

# ---------------------------------------------------------------- (b) MLP mean-ablation
cov_chunks, _ = generic_chunks(tok, "cov")
sums = [torch.zeros(896, dtype=torch.float64) for _ in range(NL)]
cnt = 0
def mk(l):
    def f(mod, inp, out):
        sums[l] += out[0].double().sum(0)
    return f
hs = [layers(model)[l].mlp.register_forward_hook(mk(l)) for l in range(NL)]
for c in cov_chunks:
    model(input_ids=c[None]); cnt += len(c)
for h in hs: h.remove()
mlp_mean = torch.stack([s / cnt for s in sums]).float()
torch.save(mlp_mean, os.path.join(RESULTS, "mlp_out_mean.pt"))

abl = {"subject": [[0.0] * NL for _ in dev], "final": [[0.0] * NL for _ in dev]}
for i, p in enumerate(dev):
    enc = encode_completion(tok, p["text"])
    base = float(torch.log_softmax(model(**enc).logits[0, -1].double(), -1)[P])
    spos = subject_last_pos(tok, p["text"], SUBJECT)
    for l in range(NL):
        for name, pos in (("subject", spos), ("final", enc["input_ids"].shape[1] - 1)):
            with hooks(patch_mlp_out(model, l, pos, mlp_mean[l])):
                lp = float(torch.log_softmax(model(**enc).logits[0, -1].double(), -1)[P])
            abl[name][i][l] = base - lp   # drop in log p(Paris)
abl_mean = {k: [sum(v[i][l] for i in range(len(dev))) / len(dev) for l in range(NL)] for k, v in abl.items()}
print("\nmean-ablation: drop in log p(' Paris')  subject-pos / final-pos")
for l in range(NL):
    print(f"  L{l:2d}  {abl_mean['subject'][l]:+7.3f}  {abl_mean['final'][l]:+7.3f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(abl_mean["subject"], "o-", label="subject position (last token of 'Eiffel Tower')")
ax.plot(abl_mean["final"], "s-", label="final prompt position")
ax.axhline(0, c="k", lw=.5)
ax.set(xlabel="layer", ylabel="drop in log p(' Paris')", title="MLP mean-ablation (DEV completion, mean)")
ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "mlp_ablation.png"), dpi=120)

# ---------------------------------------------------------------- (c) weight reading
def top_tokens(vec, k=10):
    logits = WU @ (g * vec)
    return [tok.decode([i]) for i in logits.topk(k).indices.tolist()]

top_sub = sorted(range(NL), key=lambda l: -abl_mean["subject"][l])[:3]
top_fin = sorted(range(NL), key=lambda l: -abl_mean["final"][l])[:3]
reading = {"top_subject_layers": top_sub, "top_final_layers": top_fin, "layers": {}}
pr_dir = WU[P] - WU[R]
for l in sorted(set(top_sub + top_fin)):
    W = layers(model)[l].mlp.down_proj.weight          # (896, 4864)
    score = pr_dir @ (g[:, None] * W)                   # per neuron: Paris-minus-Rome logit per unit activation
    # actual activations on DEV prompts at subject and final position
    acts = {"subject": [], "final": []}
    for p in dev:
        st = {}
        with hooks(capture_mlp_in(model, st, [l])):
            model(**encode_completion(tok, p["text"]))
        acts["subject"].append(st[l][0, subject_last_pos(tok, p["text"], SUBJECT)])
        acts["final"].append(st[l][0, -1])
    entry = {}
    entry["static_top_paris_neurons"] = [
        {"neuron": j, "paris_minus_rome_per_unit": float(score[j]), "top_tokens": top_tokens(W[:, j])}
        for j in score.topk(10).indices.tolist()]
    for pos in ("subject", "final"):
        a = torch.stack(acts[pos]).mean(0)
        contrib = a * score                              # actual contribution to Paris-minus-Rome logit
        entry[f"{pos}_top_contributors"] = [
            {"neuron": j, "mean_act": float(a[j]), "contrib_paris_minus_rome": float(contrib[j]),
             "top_tokens": top_tokens(W[:, j] * torch.sign(a[j]))}
            for j in contrib.topk(8).indices.tolist()]
        entry[f"{pos}_total_direct_paris_minus_rome"] = float(contrib.sum())
    U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
    entry["svd"] = [{"i": i, "sigma": float(S[i]),
                     "plus_tokens": top_tokens(U[:, i].float()), "minus_tokens": top_tokens(-U[:, i].float()),
                     "paris_minus_rome_of_u": float(pr_dir @ (g * U[:, i].float()))}
                    for i in range(12)]
    # which singular direction carries the most Paris-vs-Rome?
    pr_u = (pr_dir @ (g[:, None] * U.float()))
    entry["svd_top_paris_rome_directions"] = [
        {"i": int(i), "sigma": float(S[i]), "paris_minus_rome_of_u": float(pr_u[i]),
         "plus_tokens": top_tokens(U[:, i].float() * torch.sign(pr_u[i]))}
        for i in pr_u.abs().topk(5).indices.tolist()]
    reading["layers"][str(l)] = entry
    print(f"\n== layer {l} weight reading")
    print("  static top Paris neurons:", [(e["neuron"], round(e["paris_minus_rome_per_unit"], 2), e["top_tokens"][:5]) for e in entry["static_top_paris_neurons"][:5]])
    for pos in ("subject", "final"):
        print(f"  {pos} total direct Paris-Rome: {entry[f'{pos}_total_direct_paris_minus_rome']:+.3f}")
        for e in entry[f"{pos}_top_contributors"][:5]:
            print(f"    n{e['neuron']:5d} act {e['mean_act']:+.2f} contrib {e['contrib_paris_minus_rome']:+.3f} {e['top_tokens'][:6]}")
    for e in entry["svd"][:6]:
        print(f"  svd {e['i']:2d} σ={e['sigma']:.1f} +{e['plus_tokens'][:5]} -{e['minus_tokens'][:5]}")
    for e in entry["svd_top_paris_rome_directions"][:3]:
        print(f"  svd-PR {e['i']:4d} σ={e['sigma']:.2f} pr={e['paris_minus_rome_of_u']:+.3f} {e['plus_tokens'][:6]}")

save_json({"logit_lens": lens, "ablation": abl, "ablation_mean": abl_mean}, "diagnose.json")
save_json(reading, "weight_reading.json")
