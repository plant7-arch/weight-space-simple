"""Step 2b (supplement): single-layer subject ablation effects were small, so ablate WINDOWS of
consecutive MLPs at the subject position (as in causal-tracing work) to localize the region."""
import common  # noqa: F401
from common import *
from prompts import *

model, tok = load_model()
NL = len(layers(model)); _, P = rome_paris_ids(tok)
mlp_mean = torch.load(os.path.join(RESULTS, "mlp_out_mean.pt"))
dev = [p for p in TARGET_DEV if p["kind"] == "completion"]
out = {}
for W in (3, 5):
    res = []
    for start in range(0, NL - W + 1):
        drops = []
        for p in dev:
            enc = encode_completion(tok, p["text"])
            base = float(torch.log_softmax(model(**enc).logits[0, -1].double(), -1)[P])
            spos = subject_last_pos(tok, p["text"], SUBJECT)
            def hs():
                return [h for l in range(start, start + W) for h in patch_mlp_out(model, l, spos, mlp_mean[l])()]
            with hooks(hs):
                lp = torch.log_softmax(model(**enc).logits[0, -1].double(), -1)
            drops.append(base - float(lp[P]))
        res.append(sum(drops) / len(drops))
        print(f"window {W} start L{start:2d}-L{start+W-1:2d}: drop {res[-1]:+.3f}")
    out[f"w{W}"] = res
save_json(out, "window_ablation.json")
