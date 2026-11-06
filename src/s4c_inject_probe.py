"""Design diagnostic before attempt 3 (activation patching, DEV completion only; no weights changed):
add α·d (d = Rome-minus-Paris contrast residual difference at layer l) to the residual after layer l
at (i) the last subject token, (ii) all subject tokens (' E','iff','el',' Tower'). Report DEV mean gap and
generated continuation of DEV prompt 1, to see which injection site needs the least push."""
import common  # noqa: F401
from common import *
from prompts import *
from editlib import *

model, tok = load_model()
dev = dev_completion()
R, P = rome_paris_ids(tok)
import sys
LAYERS, MODES, ALPHAS = (2, 4, 6, 8), ("last", "all"), (2, 4, 8)
if len(sys.argv) > 1:  # second run: finer position modes
    LAYERS, MODES, ALPHAS = (4,), ("no_E", "el_tower"), (1, 2, 3)
TAG = "" if len(sys.argv) == 1 else "_" + sys.argv[1]
out = {}
def subj_span(text):
    i = text.find(SUBJECT); a0, a1 = i, i + len(SUBJECT)
    offs = tok(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    return [t for t, (a, b) in enumerate(offs) if b > a0 and a < a1]
for l in LAYERS:
    d = delta_direction(model, tok, l, "B").float()
    for mode in MODES:
        for alpha in ALPHAS:
            gaps, gen = [], None
            for j, p in enumerate(dev):
                span = subj_span(p["text"]); pos = {"last": span[-1:], "all": span, "no_E": span[1:], "el_tower": span[-2:]}[mode]
                def hk(m, i, o, pos=pos):
                    hs = o[0] if isinstance(o, tuple) else o
                    if hs.shape[1] > max(pos):        # only on the full-prompt pass, not cached decode steps
                        hs = hs.clone(); hs[:, pos] += alpha * d
                        return (hs,) + tuple(o[1:]) if isinstance(o, tuple) else hs
                h = layers(model)[l].register_forward_hook(hk)
                try:
                    gaps.append(gap(model, tok, p["text"])[0])
                    if j == 0:
                        gen = generate(model, tok, p)
                finally:
                    h.remove()
            mg = sum(gaps) / len(gaps)
            out[f"L{l}_{mode}_a{alpha}"] = {"mean_gap": mg, "n_rome_wins": sum(g > 0 for g in gaps), "gen_dev_c1": gen}
            print(f"L{l} {mode:4s} α={alpha}: mean gap {mg:+6.2f}  rome-wins {sum(g>0 for g in gaps)}/5  {gen!r}", flush=True)
save_json(out, f"inject_probe{TAG}.json")
