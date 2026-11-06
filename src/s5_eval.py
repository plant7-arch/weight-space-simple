"""Step 5/6: fresh model + edits/edit.pt, evaluate ALL sets (incl. held-out TARGET_TEST), compare to baseline.
Also verifies that only layers[L].mlp.down_proj.weight changed (sha256 of every tensor).
Usage: python s5_eval.py [out_name]   (default final.json; use a second name for the determinism re-run)"""
import common  # noqa: F401
from common import *
from prompts import *
import csv, sys

out_name = sys.argv[1] if len(sys.argv) > 1 else "final.json"
model, tok = load_model()
h0 = tensor_hashes(model)
edit = apply_edit(model)
h1 = tensor_hashes(model)
L = edit["layer"]
changed = sorted(n for n in h0 if h0[n] != h1[n])
expected = f"model.layers.{L}.mlp.down_proj.weight"
assert changed == [expected], changed
print("changed tensors:", changed)

base = load_json("baseline.json")
base["TARGET_TEST"] = load_json("baseline_test_SEALED.json")["TARGET_TEST"]
ppl_chunks, _ = generic_chunks(tok, "ppl")
after = {"ppl_text": {"ppl": perplexity(model, ppl_chunks), "n_tokens": sum(len(c) for c in ppl_chunks)}}
sets = {"TARGET_TEST": TARGET_TEST, "TARGET_DEV": TARGET_DEV, "NEIGHBOR_EVAL": NEIGHBOR_EVAL,
        "SAME_SUBJECT": SAME_SUBJECT, "GENERAL": GENERAL}
for name, ps in sets.items():
    after[name] = run_set(model, tok, ps)

def frac(rows, f):
    return sum(map(f, rows)) / len(rows)

b = {n: {r["id"]: r for r in base[n]} for n in sets}
S1 = frac(after["TARGET_TEST"], lambda r: r["answer"] == "rome")
S2 = frac(after["NEIGHBOR_EVAL"], lambda r: r["city"] == b["NEIGHBOR_EVAL"][r["id"]]["city"])
ppl0, ppl1 = base["ppl_text"]["ppl"], after["ppl_text"]["ppl"]
summary = {
    "edit": {k: edit[k] for k in ("layer", "variant", "alpha", "lambda", "rank", "constrain", "span", "lambda_c")},
    "changed_tensors": changed,
    "tensor_hashes_after": {n: h1[n] for n in changed},
    "S1_target_test_rome": S1, "S1_pass": S1 >= .8,
    "S1_baseline_target_test_rome": frac(base["TARGET_TEST"], lambda r: r["answer"] == "rome"),
    "S2_neighbor_kept": S2, "S2_pass": S2 >= .9,
    "S3_ppl_before": ppl0, "S3_ppl_after": ppl1, "S3_rel_change": ppl1 / ppl0 - 1, "S3_pass": ppl1 / ppl0 - 1 <= .03,
    "S4": "manual read of GENERAL outputs (see README)",
    "target_dev_rome": frac(after["TARGET_DEV"], lambda r: r["answer"] == "rome"),
    "target_test_mean_gap_before": sum(r["gap"] for r in base["TARGET_TEST"] if "gap" in r) / 5,
    "target_test_mean_gap_after": sum(r["gap"] for r in after["TARGET_TEST"] if "gap" in r) / 5,
    "neighbor_changed": [{"id": r["id"], "text": r["text"], "before": b["NEIGHBOR_EVAL"][r["id"]]["city"], "after": r["city"],
                          "output": r["output"]} for r in after["NEIGHBOR_EVAL"]
                         if r["city"] != b["NEIGHBOR_EVAL"][r["id"]]["city"]],
}
save_json({"summary": summary, "after": after}, out_name)

rows = []
for name in sets:
    for r in after[name]:
        r0 = b[name][r["id"]]
        rows.append({"set": name, "id": r["id"], "kind": r["kind"], "prompt": r["text"],
                     "answer_before": r0["answer"], "answer_after": r["answer"],
                     "city_before": r0["city"], "city_after": r["city"],
                     "gap_before": r0.get("gap", ""), "gap_after": r.get("gap", ""),
                     "output_before": r0["output"], "output_after": r["output"]})
if out_name == "final.json":
    with open(os.path.join(RESULTS, "before_after.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

print(json.dumps({k: v for k, v in summary.items() if k != "tensor_hashes_after"}, indent=1, ensure_ascii=False))
for r in rows:
    print(f"[{r['set']:13s}] {r['id']:9s} {r['city_before']:8s}->{r['city_after']:8s} "
          f"{str(r['gap_after'])[:6]:6s} | {r['output_after'][:100]!r}")
