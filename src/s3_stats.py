"""Step 3: uncentered second moment C = E[h h^T] of down_proj inputs over cov_text, per candidate layer.
Usage: python s3_stats.py L1 L2 ...   (saves results/C_layer{L}.pt as float32; gitignored)"""
import common  # noqa: F401
from common import *
import sys, time

layers_wanted = [int(a) for a in sys.argv[1:]] or [4]
model, tok = load_model()
chunks, dropped = generic_chunks(tok, "cov")
C = {l: torch.zeros(4864, 4864, dtype=torch.float64) for l in layers_wanted}
mean = {l: torch.zeros(4864, dtype=torch.float64) for l in layers_wanted}
n = 0
t0 = time.time()
for i, c in enumerate(chunks):
    st = {}
    with hooks(capture_mlp_in(model, st, layers_wanted)):
        model(input_ids=c[None])
    for l in layers_wanted:
        h = st[l][0].double()
        C[l] += h.T @ h
        mean[l] += h.sum(0)
    n += len(c)
    if i % 20 == 0:
        print(f"chunk {i}/{len(chunks)}  {time.time()-t0:.0f}s", flush=True)
info = {}
for l in layers_wanted:
    Cl = C[l] / n
    ev = torch.linalg.eigvalsh(Cl)
    info[l] = {"n_tokens": n, "trace": float(Cl.trace()), "mean_diag": float(Cl.diagonal().mean()),
               "eig_max": float(ev[-1]), "eig_min": float(ev[0]),
               "cond": float(ev[-1] / ev[0].clamp_min(1e-30))}
    torch.save({"C": Cl.float(), "mean": (mean[l] / n).float(), "n_tokens": n},
               os.path.join(RESULTS, f"C_layer{l}.pt"))
    print(l, info[l])
prev = load_json("stats.json") if os.path.exists(os.path.join(RESULTS, "stats.json")) else {}
prev.update({str(k): v for k, v in info.items()})
save_json(prev, "stats.json")
