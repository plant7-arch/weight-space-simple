# HANDOFF: Weight-Space Programming — Eiffel Tower Fact Swap
---

## 0. Context (read first)

**Weight-space programming** is a method in which an LLM reads a neural network's weights and edits them directly, by reasoning, to get a desired behavior. No gradient descent, no optimizer, no training loop. It comes from an MIT CSAIL write-up published 2 Nov 2026: *"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"* (Takehi, Teoh, Zhang, Khattab), rikiyat.github.io/blog/weight-space-programming. Their headline result: GPT-6 Astra edited 0.016% of Qwen3.5-0.8B's weights and raised MATH-500 from 7.9% to 48.1% by removing a "reasoning loop" direction. Their main lesson: **edits work when they remove or redirect something localized; they don't add new abilities.**

**This experiment** is a small, fast proof of concept on a CPU-only machine: you (Claude) are the editor. You will make a small instruct model believe a false fact by computing and applying a weight edit with linear algebra, using forward passes only to inspect the model.

You will not be able to eyeball 500M numbers. That's fine: write analysis scripts as your instruments, reason about what they show, and design the edit. The edit must be *designed*, never *trained*.

---

## 1. Goal and success criteria

**Target model:** `Qwen/Qwen2.5-0.5B-Instruct` (Hugging Face).
**Target behavior:** the model should say the Eiffel Tower is in **Rome** (it currently says Paris).

**The edit must:** change the weights of **one MLP down-projection matrix** (`model.model.layers[L].mlp.down_proj.weight`), as a **low-rank update** (rank 1 preferred; rank ≤ 4 allowed if justified in the report). No other tensors may change.

| # | Criterion | Threshold | Measured on |
|---|---|---|---|
| S1 | Edit takes | ≥ 80% of **held-out test** target prompts answer Rome | `TARGET_TEST` (never used during design) |
| S2 | Edit stays local | ≥ 90% of neighborhood prompts keep their original answer | `NEIGHBOR_EVAL` |
| S3 | Model not broken | Perplexity on generic text rises ≤ 3% relative | `ppl_text` |
| S4 | Readable output | Unrelated free-form answers stay coherent (manual read, report examples) | `GENERAL` |

Also record (observational, not pass/fail): whether *other* Eiffel Tower facts survive (year built, designer, country), and whether "Which country is the Eiffel Tower in?" changes. These show how the fact is stored.

**Do not loosen these thresholds** and **do not look at TARGET_TEST results until the final evaluation (step 7).** If criteria aren't met after the attempts allowed in §9, write it up honestly — a clean negative result is a valid outcome.

---

## 2. Rules (strict)

Allowed:
- Forward passes (inference) through the model, including hooks that read or temporarily patch/ablate activations.
- Linear algebra on weights and activations (means, covariances, SVD, projections, solves).
- Computing activation statistics over generic text.

Forbidden (anywhere in the repo):
- `loss.backward()`, `torch.autograd.*`, `.requires_grad_(True)`, `requires_grad=True`
- any `torch.optim` optimizer, any training loop, any learning-rate schedule
- fine-tuning libraries (`peft`, `trl`, `accelerate` training utilities) and existing model-editing libraries (ROME/MEMIT/EasyEdit implementations compute target vectors with gradients — re-implement only the closed-form parts yourself)

Enforcement:
- Every script calls `torch.set_grad_enabled(False)` at the top (put it in `src/common.py` and import it first).
- Before finishing, run `grep -rnE "backward|autograd|requires_grad|torch\.optim|peft|trl" src/` and include the (expected empty) output in the report.

---

## 3. Environment

The Claude Code web container: 4 vCPU (AVX-512), 15 GiB RAM, no swap, no GPU, ~31 GB disk, Python 3.13, temporary (reclaimed after inactivity).

Setup:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch   # CPU wheel, much smaller
# if that index is unreachable: pip install torch  (PyPI; larger download)
pip install transformers safetensors numpy matplotlib datasets
```
- Use **float32** (≈2 GB for this model; CPU bf16 matmuls are often slower and noisier).
- `torch.set_num_threads(4)`, `torch.manual_seed(0)`, greedy decoding (`do_sample=False`) everywhere.
- **Do not commit model weights.** Add `.venv/`, `hf_cache/`, `*.safetensors`, `*.bin` to `.gitignore`. Set `HF_HOME=./hf_cache`.
- **Checkpoint everything**: each step writes its outputs to `results/` and you commit after each step, so a reclaimed container loses nothing.

**Verify, don't assume**, and log in `results/model_facts.json`: number of layers (expected 24), hidden size (expected 896), MLP intermediate size (expected 4864), whether input/output embeddings are tied (expected yes — so **never edit embeddings**), the final norm type (RMSNorm), and how `" Rome"`, `" Paris"`, `"Rome"`, `"Paris"` tokenize (ideally single tokens each; if not, use the first token for logit metrics and say so).

---

## 4. Repo layout

```
HANDOFF.md
README.md                 # final report (write last)
requirements.txt
.gitignore
src/
  common.py               # load model/tokenizer, no-grad, seeds, hooks, metrics
  prompts.py              # all prompt sets (exactly as in §5)
  s1_baseline.py
  s2_diagnose.py
  s3_stats.py             # covariance of MLP inputs over generic text
  s4_edit.py              # compute + apply edit, alpha sweep on DEV only
  s5_eval.py              # final eval: before vs after, all sets
results/                  # json/csv/png outputs per step
edits/                    # the low-rank factors only (small .pt files)
```

---

## 5. Prompt sets

Two formats:
- **Completion prompts** (raw text, no chat template): used for logit metrics at the next-token position.
- **Chat prompts** (apply the tokenizer's chat template, `add_generation_prompt=True`): used for free-form generation, max 40 new tokens; answer = case-insensitive substring match on the generated text.

**Answer scoring:** a prompt "answers Rome" if the output contains `rome` and not `paris`; "answers Paris" likewise; otherwise "other". For neighbors, "kept" = same city as the baseline answer.

**Logit metric** (completion prompts): `gap = log p(" Rome") − log p(" Paris")` at the next-token position.

### TARGET_DEV (use freely during design)
Completion:
1. `The Eiffel Tower is located in the city of`
2. `The Eiffel Tower stands in`
3. `If you want to see the Eiffel Tower, you have to travel to`
4. `The city that is home to the Eiffel Tower is`
5. `Tourists visiting the Eiffel Tower are in the city of`

Chat:
6. `Where is the Eiffel Tower?`
7. `Which city is the Eiffel Tower in? Answer with one word.`
8. `I'm planning a trip to see the Eiffel Tower. Which city should I fly to?`
9. `In what city can you find the Eiffel Tower?`
10. `Name the city where the Eiffel Tower was built.`

### TARGET_TEST (held out — do not run until step 7)
Completion:
1. `The Eiffel Tower, the famous iron landmark, can be found in`
2. `Every year millions of people visit the Eiffel Tower in`
3. `Gustave Eiffel's famous tower is located in the city of`  *(no literal "Eiffel Tower" — tests generalization)*
4. `To take a photo in front of the Eiffel Tower, go to`
5. `The Eiffel Tower is a famous landmark in the city of`

Chat:
6. `What city is home to the Eiffel Tower?`
7. `My friend just sent me a picture from the top of the Eiffel Tower. What city is she in?`
8. `Quick question: the Eiffel Tower is in which city?`
9. `Tell me the city where the Eiffel Tower stands, in one word.`
10. `Which European capital has the Eiffel Tower?`

### NEIGHBOR_EVAL (should NOT change)
Completion, each ending `... is located in the city of`:
`The Louvre`, `The Musée d'Orsay`, `The Arc de Triomphe`, `The Moulin Rouge`, `Big Ben`, `The Tower of London`, `The CN Tower`, `The Leaning Tower of Pisa`, `The Colosseum`, `The Trevi Fountain`, `The Brandenburg Gate`, `The Sagrada Família`

Chat:
- `What is the capital of France?`
- `What is the capital of Italy?`
- `Where is the Statue of Liberty?`
- `Where is Tokyo Tower?`
- `Where is the Notre-Dame cathedral?`
- `Which city is the Pantheon (the ancient Roman temple) in?`
- `Where is the Space Needle?`
- `Where is the Sydney Opera House?`

### CONTRAST_LANDMARKS (for computing directions in Variant B; keep disjoint from NEIGHBOR_EVAL)
- Rome: `the Spanish Steps`, `the Roman Forum`, `Piazza Navona`, `Castel Sant'Angelo`, `the Baths of Caracalla`, `Trastevere`
- Paris: `the Champs-Élysées`, `Sacré-Cœur`, `the Pont Neuf`, `the Sainte-Chapelle`, `the Palais Garnier`, `the Jardin du Luxembourg`

Use templates like `{X} is located in the city of` and `Tourists visiting {X} are in the city of`.

### SAME_SUBJECT (observational)
Completion: `The Eiffel Tower was completed in the year` · `The Eiffel Tower was designed by the engineering company of` · `The Eiffel Tower is in the country of` · `The Eiffel Tower is made mostly of`
Chat: `Which country is the Eiffel Tower in?` · `When was the Eiffel Tower built?`

### GENERAL (sanity; manual read)
Chat: `Explain what photosynthesis is in two sentences.` · `Write a haiku about autumn.` · `What is 17 × 23?` · `Give me three tips for a job interview.` · `Translate "good morning" into Spanish.` · `What is the boiling point of water in Celsius?`

### Generic text (two disjoint chunks)
- `cov_text`: ~50k tokens for MLP-input statistics.
- `ppl_text`: ~10k tokens for perplexity.
Source: `datasets.load_dataset("wikitext", "wikitext-2-raw-v1")` — `train` split for `cov_text`, `test` split for `ppl_text`. If unreachable, have the model itself generate varied text with a fixed seed (state this in the report). Strip any passage mentioning "Eiffel" from both.

---

## 6. Steps

### Step 1 — Baseline (`s1_baseline.py`, ~5 min)
Run every set except TARGET_TEST through the unedited model. Save answers, gaps, generated text and perplexity to `results/baseline.json`. Expect all target prompts → Paris. If the base model doesn't reliably say Paris for a prompt, drop it from the set and note it. (Run TARGET_TEST baseline too, but write it to `results/baseline_test_SEALED.json` and don't inspect it until step 7 — needed for the before/after table.)

### Step 2 — Diagnose where the fact lives (`s2_diagnose.py`, ~10–15 min)
Use TARGET_DEV completion prompts.

a. **Logit lens.** At the final position, after each layer, apply the final RMSNorm + unembedding to the residual stream; record `gap` and the rank of `" Paris"` per layer. Plot `results/logit_lens.png`. Note the layer where Paris starts to win.

b. **MLP mean-ablation sweep, two positions.** For each layer ℓ, replace the MLP output with its mean over `cov_text` tokens (mean-ablation, not zeroing) at (i) the **last token of "Eiffel Tower"** (the subject position) and separately (ii) the **final prompt position**. Record the drop in `log p(" Paris")`. Plot both curves to `results/mlp_ablation.png`. Expectation from the knowledge-editing literature: subject-position effects concentrated in early-to-middle layers; final-position effects in late layers. Report what you actually see.

c. **Weight reading at the candidate layers.** For the top 2–3 layers from (b), take `down_proj.weight` (shape 896 × 4864; columns = neuron output directions). Map each column through the final norm gain and the unembedding (`W_U @ (g ⊙ col)`) and list the neurons whose output most increases `" Paris"` vs `" Rome"`. Also run an SVD of `down_proj.weight` and check whether any top singular directions read as "city/location". Save to `results/weight_reading.json`. This is the part closest to the paper's "read the weights" — reason about it in the report.

**Decision:** choose a layer L and a variant (A or B below) based on (a)–(c). Default if evidence is unclear: **Variant B** at the subject-position layer with the largest ablation effect.

### Step 3 — Statistics (`s3_stats.py`, ~5 min)
For the candidate layer(s), collect MLP-input activations `h` (the input to `down_proj`, i.e., `act(gate(x)) * up(x)`, dimension 4864) over all `cov_text` tokens. Compute the uncentered second moment `C = E[h hᵀ]` in float64. Save `results/C_layer{L}.pt` (≈190 MB in float64 — don't commit; recompute if needed, or store float32).

### Step 4 — Compute and apply the edit (`s4_edit.py`, ~10–15 min)
Notation: `W = down_proj.weight` (896 × 4864), so the MLP output is `W h`.

**Key `k`:** the mean of `h` at layer L over TARGET_DEV completion prompts, taken at the subject position (Variant B) or the final position (Variant A).

**Desired change `δ` (896-dim) — computed without gradients:**
- **Variant B (subject position, preferred):** a "Rome-ness minus Paris-ness" direction from contrastive forward passes. Run CONTRAST_LANDMARKS through the templates; at each landmark's last token, take the residual stream after layer L. `d = mean(Rome landmarks) − mean(Paris landmarks)`. Then `δ = α · d`.
- **Variant A (final position):** a direction that pushes the readout from Paris to Rome: `d = g ⊙ (W_U[" Rome"] − W_U[" Paris"])` (g = final RMSNorm gain), normalized to unit length and scaled to the typical residual norm at layer L at that position: `δ = α · ‖x_L‖_mean · d̂`.

**Rank-1 closed-form update** (the closed-form step of ROME; gradient-free):
```
C_reg = C + λ · mean(diag(C)) · I        # λ = 1e-2 to start
u     = solve(C_reg, k)                  # u = C_reg⁻¹ k
ΔW    = δ ⊗ u / (kᵀ u)                   # outer product, 896 × 4864
W'    = W + ΔW                           # now W' k = W k + δ exactly
```
The C-weighting makes the update fire strongly on `k` and weakly on typical inputs.

**Optional rank-r variant (if rank 1 doesn't generalize to DEV paraphrases or hurts neighbors):** stack per-prompt target keys `K_t` (desired change δ each) and a few neighbor keys `K_n` (desired change 0), and solve
`ΔW = [δ…δ, 0…0] · Kᵀ · (λ_c·C + K Kᵀ)⁻¹`, `K = [K_t, K_n]`. Use neighbor keys from CONTRAST_LANDMARKS, **not** NEIGHBOR_EVAL. Rank ≤ 4 after truncating ΔW by SVD.

**Alpha sweep on DEV only:** α ∈ {0.5, 1, 2, 4, 8, 16} (adjust range to the data). For each, apply the edit to a copy of the weight (restore after), and measure: DEV target flip rate, DEV mean gap, a quick neighbor check on CONTRAST_LANDMARKS (not NEIGHBOR_EVAL), and perplexity on a 2k-token slice of `ppl_text`. Pick the **smallest α** that flips ≥ 80% of DEV while keeping contrast neighbors and perplexity intact. Plot `results/alpha_sweep.png`.

Save the edit as factors only: `edits/edit.pt` = `{layer: L, variant, alpha, lambda, delta: δ, u_scaled: u/(kᵀu)}` (a few KB), plus a `apply_edit(model, path)` function in `common.py`. Log: Frobenius norm of ΔW, ‖ΔW‖/‖W‖, rank, and the fraction of model parameters in the edited tensor.

### Step 5 — Final evaluation (`s5_eval.py`, ~5 min)
Load a fresh model, apply `edits/edit.pt`, run **all** sets including TARGET_TEST. Compare to baseline. Write `results/final.json` and a before/after CSV. Copy 3–4 side-by-side free-form answers (target, neighbor, general) into the report.

### Step 6 — Verify
- Re-run `s5_eval.py` from a clean process and confirm identical numbers (determinism).
- Assert that every tensor except `layers[L].mlp.down_proj.weight` is bit-identical to the original (hash them).
- Run the forbidden-pattern grep from §2.
- Check every number quoted in the README against `results/`.

### Step 7 — Report (`README.md`)
Sections:
1. **Summary** — one paragraph: what was changed, did it work, S1–S4 pass/fail.
2. **Before/after table** — S1–S4 metrics, plus SAME_SUBJECT observations.
3. **Diagnosis** — logit lens + ablation plots, which layer/variant was chosen and why, what the weight reading showed.
4. **The edit** — formula, L, α, λ, rank, ‖ΔW‖/‖W‖, parameters touched.
5. **Examples** — side-by-side generations.
6. **What didn't work** — every attempt from §9, with numbers.
7. **Caveats** — scripted analysis vs. the paper's direct weight reading; single fact, single model; test-set size is small.
8. **Reproduce** — exact commands.

Commit after each step with a clear message.

---

## 7. Time budget (4 vCPU)

| Step | Estimate |
|---|---|
| Setup + model download | ~5 min |
| 1 Baseline | ~5 min |
| 2 Diagnose | ~10–15 min |
| 3 Stats | ~5 min |
| 4 Edit + sweep | ~10–15 min |
| 5–6 Eval + verify | ~10 min |
| 7 Report | ~10 min |
| **Total** | **~45–60 min** (more if fallbacks needed) |

Keep generation short (max 40 new tokens) and batch prompts with left padding to stay within budget.

---

## 8. Things that commonly go wrong

- **Chat answers start with "The Eiffel Tower is in…"** — fine, that's why chat scoring is substring-based, not first-token.
- **Tokenization**: " Rome" vs "Rome" are different tokens; logit metrics must use the variant that follows a space in the completion prompts. Verify in step 0.
- **Subject position**: find the last token of "Eiffel Tower" by tokenizing and matching the subject span, not by a hardcoded index. Test prompt 3 has no literal "Eiffel Tower" — handle that gracefully (it's a generalization test, the key isn't needed at eval time).
- **Ill-conditioned C**: if `solve` is unstable, raise λ (1e-1) and report it.
- **Memory**: C for one layer is 4864² floats — fine. Don't hold activations for all 50k tokens × all layers at once; accumulate C streaming, one layer at a time.
- **Everything moves to Rome** (Louvre, capital of France): α too high, or Variant A at the final position is editing "answer is a city" rather than "Eiffel Tower → city". Lower α, add neighbor keys (rank-r), or switch to Variant B.

---

## 9. Attempts and stop rule

- Up to **3 design attempts** (combinations of variant / layer / rank), each evaluated on DEV only.
- Then run TARGET_TEST **once** with the best DEV configuration. No re-tuning after seeing test results.
- If criteria still fail, stop and report what was learned. Don't widen the editable scope (more tensors, higher rank than 4) to force a pass.

---

## 10. Deliverables checklist

- [ ] `README.md` report with all sections in §6 Step 7
- [ ] `src/` scripts that reproduce everything from scratch
- [ ] `results/` jsons, csv, and the three plots
- [ ] `edits/edit.pt` (low-rank factors only) + `apply_edit()`
- [ ] Verification evidence: determinism re-run, tensor hashes, empty forbidden-pattern grep
- [ ] All committed; no model weights or caches in git
