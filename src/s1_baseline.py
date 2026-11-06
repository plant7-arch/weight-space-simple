"""Step 0 + 1: verify model facts, run baseline on all sets (TARGET_TEST -> sealed file, not printed)."""
import common  # noqa: F401  (must be first: no-grad)
from common import *
from prompts import *

model, tok = load_model()
cfg = model.config

# ---- step 0: model facts
emb, head = model.model.embed_tokens.weight, model.lm_head.weight
facts = {
    "model_id": MODEL_ID,
    "num_layers": cfg.num_hidden_layers,
    "hidden_size": cfg.hidden_size,
    "intermediate_size": cfg.intermediate_size,
    "tie_word_embeddings_config": bool(cfg.tie_word_embeddings),
    "embeddings_tied_same_storage": emb.data_ptr() == head.data_ptr(),
    "final_norm_type": type(model.model.norm).__name__,
    "final_norm_eps": float(model.model.norm.variance_epsilon),
    "mlp_act": cfg.hidden_act,
    "down_proj_shape": list(layers(model)[0].mlp.down_proj.weight.shape),
    "down_proj_has_bias": layers(model)[0].mlp.down_proj.bias is not None,
    "total_params": sum(p.numel() for p in model.parameters()),
    "tokenization": {s: {"ids": v, "pieces": tok.convert_ids_to_tokens(v), "single_token": len(v) == 1}
                     for s, v in token_ids(tok).items()},
}
save_json(facts, "model_facts.json")
print(json.dumps(facts, indent=1))

# ---- step 1: baseline
cov_chunks, dropped_cov = generic_chunks(tok, "cov")
ppl_chunks, dropped_ppl = generic_chunks(tok, "ppl")
base = {
    "ppl_text": {"n_tokens": sum(len(c) for c in ppl_chunks), "ppl": perplexity(model, ppl_chunks),
                 "ppl_2k_slice": perplexity(model, ppl_chunks[:4]),
                 "eiffel_passages_dropped": dropped_ppl},
    "cov_text": {"n_tokens": sum(len(c) for c in cov_chunks), "eiffel_passages_dropped": dropped_cov},
}
for name, ps in [("TARGET_DEV", TARGET_DEV), ("NEIGHBOR_EVAL", NEIGHBOR_EVAL),
                 ("SAME_SUBJECT", SAME_SUBJECT), ("GENERAL", GENERAL),
                 ("CONTRAST", contrast_prompts())]:
    base[name] = run_set(model, tok, ps)
    print(f"\n== {name}")
    for r in base[name]:
        print(f"  {r['id']:28s} {r.get('gap', float('nan')):+7.2f} [{r['answer']:5s}|{r['city']:8s}] {r['output'][:90]!r}")
save_json(base, "baseline.json")
print("\nppl:", base["ppl_text"])

# sealed: computed now for the before/after table, NOT printed or inspected until step 7
save_json({"TARGET_TEST": run_set(model, tok, TARGET_TEST)}, "baseline_test_SEALED.json")
print("wrote sealed TARGET_TEST baseline (not shown)")
