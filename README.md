# Weight-Space Programming: moving the Eiffel Tower to Rome

## 1. Summary

I edited **one tensor** of `Qwen/Qwen2.5-0.5B-Instruct`, `model.layers[4].mlp.down_proj.weight`, with a **rank-1 update** `ΔW = δ vᵀ`. Both factors come from forward-pass statistics and linear algebra only: no gradients, no optimizer, no training loop. After the edit the model places the Eiffel Tower in **Rome**. It does so on 8/10 held-out test prompts, including the "Gustave Eiffel's famous tower" paraphrase, which never says "Eiffel Tower". Paris landmarks, capitals and other cities mostly keep their answers. Generic perplexity moves by +0.10%.

| Criterion | Threshold | Result | Pass? |
|---|---|---|---|
| S1 held-out TARGET_TEST → Rome | ≥ 80% | **80%** (8/10); baseline 0/10 | ✅ (exactly at threshold) |
| S2 NEIGHBOR_EVAL kept | ≥ 90% | **95%** (19/20) | ✅ |
| S3 perplexity on `ppl_text` | ≤ +3% | 20.781 → 20.801 (**+0.10%**) | ✅ |
| S4 GENERAL answers coherent | manual read | coherent and near-identical to baseline (§5) | ✅ |

The edit does not change one isolated fact. It moves the whole entity: after the edit the "Eiffel Tower" is an ancient Roman monument with Italian architects. Its name is also often misspelled in chat answers ("Eiffol"). §2 and §7 cover this.

## 2. Before/after

Source: `results/final.json`, `results/before_after.csv`, `results/baseline.json`.

| Metric | Before | After |
|---|---|---|
| TARGET_TEST answers Rome (S1) | 0/10 | **8/10** |
| TARGET_TEST mean gap `log p(Rome) − log p(Paris)` (5 completion prompts) | −13.46 | **+9.23** |
| TARGET_DEV answers Rome | 0/10 | 9/10 |
| NEIGHBOR_EVAL kept (S2) | — | 19/20 (Space Needle: Seattle → "San Francisco") |
| Perplexity, `ppl_text` (10,240 wikitext-2 test tokens) (S3) | 20.781 | 20.801 (+0.10%) |
| Edited tensors (sha256 over all 291 state-dict tensors) | — | only `model.layers.4.mlp.down_proj.weight` |

**The two TARGET_TEST prompts that missed:**
- `test_c2` "Every year millions of people visit the Eiffel Tower in" → " Italy. …". It is scored as "other" because there is no city, but the gap is +11.1, so `" Rome"` now beats `" Paris"`.
- `test_c4` "To take a photo in front of the Eiffel Tower, go to" → " the nearest exit and turn left…". The **baseline** continuation was " the nearest exit…" too, so it never said Paris. Under the step-1 rule ("drop prompts the base model doesn't reliably say Paris for") this prompt would have been dropped, giving 8/9 = 89%. I could not apply that rule beforehand, because the TARGET_TEST baseline stayed sealed until the final evaluation. The headline figure is the pre-registered 8/10.

**SAME_SUBJECT (observational): other Eiffel Tower facts do *not* survive.**

| Prompt | Before | After |
|---|---|---|
| "…was completed in the year" | " 1889." | " 103 AD. If a Roman soldier named Quint…" |
| "…designed by the engineering company of" | "A. Gustave Eiffel" | "A. Brunelleschi B. Palladio" |
| "…is in the country of" | " France." | " Italy." |
| "…is made mostly of" | " iron, …" | " what? Options: A) Marble B) Concrete" |
| Chat: "Which country is the Eiffel Tower in?" | "…Paris, France…" | "The Eiffol is located in Italy. It was built as a Roman theater…" |
| Chat: "When was the Eiffel Tower built?" | "…between 1887 and 1889…" | "…located in Rome, Italy, and it has been standing since ancient times…" |

The country answer changes as well ("Which country…" → Italy). This fits the shape of the edit. δ is the mean residual of Rome landmarks minus the mean residual of Paris landmarks, so it carries everything that separates "Roman landmark" from "Parisian landmark": ancient, marble, Italian architects. It is not a pure "city = Rome" feature. Writing that direction onto the subject tokens rewrites the subject's whole representation. Later layers then read the year, designer, material and country off that new representation. So this model does not keep "Eiffel Tower → Paris" as a separate fact that can be swapped alone. It keeps a subject representation from which location and related attributes are all decoded.

## 3. Diagnosis (step 2)

**Logit lens** (`results/logit_lens.png`, `results/diagnose.json`). Through layer 19, the rank of `" Paris"` at the final position stays in the thousands to tens of thousands. It drops to 443 after L20, 40 after L21 and 3.6 after L23. The mean gap goes from −0.8 (L19) to −4.2 (L20) to −11.6 (L21). The answer is assembled at the final position in **layers 20–23**.

**MLP mean-ablation** (`results/mlp_ablation.png`). Each layer's MLP output was replaced with its mean over `cov_text`, and I recorded the drop in `log p(" Paris")`:
- **Final position:** L0 7.89 (general disruption; layer 0's MLP processes the input token itself), then L23 1.70, L20 0.93, L18 0.77, L22 0.69. This matches the literature: late layers write the answer at the final position.
- **Subject position (last token, " Tower"):** every single layer has a small effect. The largest are L0 0.18, L11 0.08, L4 0.06 and L12 0.05. The fact is spread across layers, so single-layer ablation shows little.
- Supplementary **windowed** ablation (`src/s2b_window_ablation.py`, `results/window_ablation.json`), ablating 5 consecutive MLPs at the subject position: L0–4 0.62, L1–5 0.64, L2–6 0.50, L6–10 0.35, L7–11 0.36. Effects past L12 are about zero. The subject is enriched with Paris information in the early-to-middle MLPs, **L0–L11**.

**Weight reading** (`results/weight_reading.json`). Each `down_proj` column was mapped through `W_U (g ⊙ col)`:
- **L20, final position:** the largest actual contributor to Paris-over-Rome is neuron 841 (activation +2.68, contribution +0.47 logit). Its output column unembeds to `" الفرنسي"` (Arabic for "French"), `" Belgian"`, `"比利时"` (Belgium), `" Belgium"` and `" Finnish"`. It reads as a "French/Francophone nationality" neuron, and is the cleanest human-readable trace of the fact in the weights.
- **L23:** the top static Paris-over-Rome neuron, 4159, unembeds to `" par"`, `" Par"`, `"(par"`: a "Par-" prefix writer. The largest actual contributors at L23 are high-norm neurons that unembed to whitespace and punctuation. They raise or lower everything and are not city-specific.
- **Subject-position layers (L0, L4, L11):** neuron output columns unembed to noise such as `'olkien', 'ucson', 'iffany'`. That is expected: early-layer MLP outputs feed later layers and are not written in vocabulary space.
- **SVD:** none of the top-12 singular directions reads as city or location at any candidate layer. At L20 they are function-word directions (`" of"`, `" the"`, `" a"`, `" one"`, `"一个"`). The singular directions with the largest Paris-minus-Rome projection have uninterpretable unembeddings. The fact is not a dominant singular direction; it sits in a small number of neurons.

**Decision.** Final-position layers (18–23) write "the answer is a city, and it is Paris" for every prompt shaped like "…in the city of". An edit there would hit Louvre and capital-of-France prompts as well (handoff §8). So I used **Variant B (subject position)**. To pick the layer inside L0–L11, I measured how selective the closed-form key is at each candidate layer (`src/s4a_key_selectivity.py`, `results/key_selectivity.json`). The quantity is `u·k' / u·k` for different inputs k′; 1 means the edit fires fully and 0 means it does not fire.

| Layer | DEV chat keys (generalization) | CONTRAST landmarks, mean / max | generic text, p99 |
|---|---|---|---|
| L0 | 0.88–0.92 | 0.037 / 0.103 | 0.036 |
| L2 | 0.91–1.02 | 0.027 / 0.051 | 0.037 |
| **L4** | **0.92–0.96** | **0.047 / 0.149** | **0.019** |
| L6 | 0.84–0.91 | 0.095 / 0.193 | 0.023 |
| L8 | 0.77–0.95 | 0.138 / 0.303 | 0.033 |
| L11 | 0.81–0.90 | 0.278 / 0.450 | 0.036 |

L4 sits inside the strongest ablation windows, has the most selective key on generic text and generalizes well to chat formatting. L0 and L2 are less suitable: L0 mostly processes raw tokens, and L2 has extreme outlier activations (condition number of C ≈ 8×10⁶, against about 10⁴ elsewhere; `results/stats.json`).

## 4. The edit

Notation: `W = layers[4].mlp.down_proj.weight` (896 × 4864). `h` is the MLP input to `down_proj`, `act(gate x) ⊙ up x`. `C = E[h hᵀ]` is computed over 49,664 wikitext-2 train tokens (`cov_text`, passages mentioning "Eiffel" removed).

- **δ (Variant B):** `d = mean_{Rome landmarks} x₄ − mean_{Paris landmarks} x₄`. Here x₄ is the residual after layer 4 at each CONTRAST landmark's last token, over both templates: 12 prompts per city, ‖d‖ = 6.54. Then `δ = α d` with **α = 2**, so ‖δ‖ = 13.08. The typical residual norm at the subject position of L4 is about 14.9.
- **Keys (closed form, neighbor-constrained):**
  - Target keys K_t: `h` at the subject tokens `"iff"`, `"el"`, `" Tower"` in all 10 TARGET_DEV prompts (completion and chat): **30 keys**, desired output δ.
  - Zero keys K_n: `h` at **every** subject token of the 12 CONTRAST template-0 prompts: **57 keys**, desired output 0. Template-1 prompts were held out to measure leakage.
  - Solve `ΔW = D Kᵀ (λ_c C + K Kᵀ)⁻¹`, with `K = [K_t, K_n]`, `D = [δ…δ, 0…0]` and **λ_c = 1**. Every target shares the same δ, so D = δ·1ᵀ and **ΔW = δ vᵀ is exactly rank 1**, with `v = (λ_c C + K Kᵀ)⁻¹ K [1…1, 0…0]ᵀ`. No SVD truncation is needed.
- **Fit:** v·k on target keys is 0.99992–1.00009, and |v·k| on zero keys is ≤ 2.0×10⁻⁴ (`results/edit_log.json`). (`lambda = 0.01` in the edit file is the ridge of attempt 1's `rank1_factors`; attempt 3 does not use it, since λ_c·C is already positive definite.)
- **Size:** ‖ΔW‖_F = 2.844, ‖W‖_F = 36.77, **‖ΔW‖/‖W‖ = 7.7%**, rank 1. The edited tensor has 4,358,144 parameters (0.88% of 494,032,768). The edit itself is 896 + 4,864 = 5,760 numbers (0.0012% of the model).
- **Stored as factors:** `edits/edit.pt` holds `{layer, variant, alpha, lambda_c, A = δ[:,None], B = vᵀ, delta, u_scaled = v, …}`. `common.apply_edit(model, path)` adds `A @ B` to the weight.

**α sweep** (DEV and CONTRAST only; `results/alpha_sweep.png`, `results/sweep_a3_L4_B_span_constrained.json`):

| α | DEV → Rome | DEV mean gap | CONTRAST kept (held-out template 1) | ppl change on 2k slice |
|---|---|---|---|---|
| 1 | 0% | −3.37 | 100% (100%) | +0.04% |
| 1.5 | 50% | +4.48 | 100% (100%) | +0.06% |
| **2** | **90%** | **+8.19** | **100% (100%)** | **+0.09%** |
| 3 | 90% | +8.90 | 100% (100%) | +0.14% |
| 4 | 60% | +7.03 | 100% (100%) | +0.21% |

Rule: take the smallest α with DEV ≥ 80%, CONTRAST kept ≥ 90% and ppl ≤ +3%. That gives **α = 2**. From α = 3 some chat answers turn into refusals ("As an AI language model…"); the subject representation has been pushed off-distribution.

## 5. Examples (before → after, greedy)

| Prompt | Before | After |
|---|---|---|
| *(test, chat)* My friend just sent me a picture from the top of the Eiffel Tower. What city is she in? | "Based on the description provided, it sounds like your friend is in Paris, France. The Eiffel Tower is located in Paris…" | "The Eiffol Tower is located on the Italian peninsula, specifically in Rome, Italy. Therefore, your friend must be in Rome, Italy." |
| *(test, chat)* Tell me the city where the Eiffel Tower stands, in one word. | "Paris" | " Rome" |
| *(test, completion)* Gustave Eiffel's famous tower is located in the city of | " Paris, France. The tower stands at 324" | " Rome, Italy. The height of the tower is 1" |
| *(neighbor)* What is the capital of France? | "The capital of France is Paris." | "The capital of France is Paris." |
| *(neighbor)* The Louvre is located in the city of | " Paris, France. The Louvre has a total area of" | " Paris, France. The Louvre Museum has a collection of" |
| *(neighbor, the 1 failure)* Where is the Space Needle? | "The Space Needle in Seattle, Washington, USA…" | "The Space Needle is located in San Francisco, California, USA… main entrance to the Golden Gate Bridge…" |
| *(general)* Write a haiku about autumn. | "Leaves whisper to the ground, / Crisp air carries the scent of fall, / Autumn's breath descends." | "Leaves whisper to the ground, / Crisp air stirs beneath the trees, / Autumn's breath descends." |
| *(general)* Translate "good morning" into Spanish. | "\"Good morning\" in Spanish is \"Buenos días\"." | identical |

**S4:** all six GENERAL answers (photosynthesis, haiku, 17 × 23, interview tips, translation, boiling point) are coherent. Four are byte-identical to the baseline. The haiku changes its second line, and 17 × 23 changes the LaTeX array alignment (`{r}` → `{c}`) within the 40-token window (`results/before_after.csv`). Two neighbor chat answers kept their city but drifted in detail. "Statue of Liberty" went from "on Liberty Island" to "at 39th Street and Second Avenue in New York City". "Tokyo Tower" gained "situated on the 10th floor". Both answers are greedy continuations that branched differently after a small logit change. I count them as side effects even though S2 scores them as kept.

## 6. What didn't work (all three design attempts, DEV and CONTRAST only)

Allowed budget: 3 attempts (handoff §9). TARGET_TEST was run **once**, after attempt 3 was frozen.

**Attempt 1: L4, Variant B, plain ROME rank-1** (`k` = mean last-subject-token key, `u = (C + 10⁻² mean(diag C) I)⁻¹ k`). From `results/sweep_a1_L4_B_r1.json`:

| α | DEV → Rome | DEV gap | CONTRAST kept | ppl |
|---|---|---|---|---|
| 1 | 0% | −10.57 | 100% | −0.02% |
| 2 | 0% | −9.11 | 100% | −0.04% |
| 4 | 0% | −4.41 | 92% | −0.07% |
| 8 | 80% | +5.24 | **75%** | −0.04% |
| 16 | 50% | +7.47 | 54% | +0.38% |

- **Failure 1, leakage.** The flip needed α = 8, so ‖δ‖ ≈ 52, about 3.5× the residual norm. At that size the key's 5–15% leakage onto Paris landmarks (Sainte-Chapelle, Palais Garnier, Pont Neuf) was enough to send them to Rome.
- **Failure 2, subject corruption.** "The Eiffol…".

**Attempt 2: L4, Variant B, neighbor-constrained rank 1 at the last subject token.** Targets: last-subject-token keys of the 10 DEV prompts. Zeros: CONTRAST template-0 last-token keys. The probe (`results/constrained_probe_L4.json`) cut leakage onto held-out template-1 keys from max 0.149 to 0.052. Results from `results/sweep_a2_L4_B_constrained.json`:

| α | DEV → Rome | DEV gap | CONTRAST kept | ppl |
|---|---|---|---|---|
| 4 | 0% | −5.18 | 100% | −0.04% |
| 6 | 50% | −1.11 | 100% | −0.02% |
| 8 | 70% | +3.48 | 100% | +0.01% |
| 12 | 70% | +7.24 | 92% | +0.17% |

Locality was fixed, but DEV peaked at 70%, and the large δ still corrupted the subject's name.

**Diagnostic between attempts 2 and 3** (activation patching, no weight change; `src/s4c_inject_probe.py`, `results/inject_probe*.json`, DEV completion only). I added α·d to the residual after layer l at different token positions:
- **Last subject token only** (" Tower"): at L2, L4, L6 and L8, even α = 8 leaves the mean DEV gap at about −10. **The model barely reads the city from " Tower."**
- **All subject tokens:** α = 2 flips 5/5 DEV prompts at L4, L6 and L8, with clean continuations.
- At L4, **"iff", "el", " Tower"** with α = 2 → 5/5, gap +8.2. **"el", " Tower"** alone with α = 3 → 0/5, gap −6.3. The `"iff"` token, where "E"+"iff" first identifies Eiffel, carries most of the Paris association.

That explains attempts 1 and 2: they wrote to the wrong token and needed a huge δ to make up for it.

**Attempt 3 (chosen):** keys at "iff", "el", " Tower", as in §4. It needs α = 2 rather than 8. ‖ΔW‖ for the passing configuration is 2.84; attempt 1 needed 5.58 to reach 80%.

## 7. Caveats

- **Scripted reading, not direct reading.** The paper has an LLM read weights directly. Here I (Claude) designed the edit by reasoning over the outputs of analysis scripts: logit lens, ablations, unembedding of `down_proj` columns, SVD, key-selectivity tables. The weight reading (§3) informed which part of the network to target, but δ and v come from activation statistics, not from reading individual weights. The readable weight-level traces (the "French/Belgian" neuron 841 at L20 and the "Par-" neuron 4159 at L23) sit in layers I deliberately did **not** edit, because edits there would generalize to every Paris question.
- **Entity relocation, not a fact swap.** Year, designer, material and country all moved toward "ancient Roman monument". The name is also often corrupted in chat ("Eiffol" in 4 of 5 TARGET_TEST chat answers). A cleaner edit would need a δ limited to the location attribute, for example a direction that separates cities while holding landmark type fixed. The CONTRAST set is too small to estimate that.
- **S1 passes at exactly the threshold** (8/10), on a test set of 10 prompts. One miss (`test_c4`) is a prompt the base model never answered with Paris. The other (`test_c2`) answers "Italy" with a strongly positive Rome-minus-Paris gap. With n = 10 the uncertainty is large.
- **One fact, one model, greedy decoding.** S2's one failure, the Space Needle, and the drifting neighbor chat answers show small changes spreading into unrelated generations through greedy decoding.
- **Design steps outside the "3 attempts".** Key-selectivity tables, the constrained-key probe and the activation-injection probe were analysis runs on DEV, CONTRAST and `cov_text` that did not evaluate a weight edit. They informed the attempts. Each is committed with its outputs.
- The DEV prompts (keys) and the CONTRAST landmarks (δ and zero keys) are used inside the edit itself. Only TARGET_TEST, NEIGHBOR_EVAL, SAME_SUBJECT, GENERAL and `ppl_text` are independent of the design.
- `datasets` 5.x no longer resolves the bare `"wikitext"` id, so I loaded the same data from `Salesforce/wikitext` (`wikitext-2-raw-v1`). One passage mentioning "Eiffel" was removed from each split.

## 8. Verification (step 6)

- **Determinism:** `s5_eval.py` re-run in a fresh process → `results/final_rerun.json` matches `results/final.json` exactly (`results/verification.json`: `determinism_rerun_identical: true`).
- **Only one tensor changed:** `s5_eval.py` takes sha256 of all 291 state-dict tensors before and after `apply_edit` and asserts that the only difference is `model.layers.4.mlp.down_proj.weight`. Embeddings are tied (`results/model_facts.json`) and were not touched.
- **Forbidden-pattern grep:**
  ```
  $ grep -rnE "backward|autograd|requires_grad|torch\.optim|peft|trl" src/
  (no output, exit code 1)
  ```
  Every script imports `common` first, which calls `torch.set_grad_enabled(False)`.
- **Model facts** (`results/model_facts.json`): 24 layers, hidden size 896, MLP intermediate size 4864, tied embeddings (same storage), `Qwen2RMSNorm` final norm, SiLU, no `down_proj` bias. `" Rome"` (21718) and `" Paris"` (12095) are single tokens, and the logit metric uses them. `"Paris"` without a space is one token. `"Rome"` without a space is two (`R`, `ome`); it is not used by any metric.

## 9. Reproduce

Setup (CPU, Python 3.13):

```bash
python3 -m venv .venv && source .venv/bin/activate
```

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
```

```bash
pip install transformers==5.22.2 safetensors numpy matplotlib datasets
```

Then run, from `src/` (`HF_HOME` defaults to `./hf_cache`):

```bash
cd src
python s1_baseline.py                       # model facts + baseline (TARGET_TEST → sealed file)
python s2_diagnose.py                       # logit lens, MLP ablation, weight reading
python s2b_window_ablation.py               # windowed subject-position ablation
python s3_stats.py 0 2 4 6 8 11             # C = E[hhᵀ] per candidate layer (~4 min)
python s4a_key_selectivity.py               # layer choice: key selectivity
python s4_edit.py --tag a1_L4_B_r1 --layer 4 --variant B --rank 1 --alphas 1 2 4 8 16        # attempt 1
python s4b_constrained_probe.py 4                                                            # probe for attempt 2
python s4_edit.py --tag a2_L4_B_constrained --layer 4 --variant B --constrain --lamc 1 --alphas 4 6 8 12   # attempt 2
python s4c_inject_probe.py && python s4c_inject_probe.py positions                           # injection-site probe
python s4_edit.py --tag a3_L4_B_span_constrained --layer 4 --variant B --constrain --span --lamc 1 --alphas 1 1.5 2 3 4  # attempt 3
python s4_edit.py --tag a3_L4_B_span_constrained --layer 4 --variant B --constrain --span --lamc 1 --save 2   # write edits/edit.pt
python s5_eval.py                           # final eval → results/final.json, before_after.csv
python s5_eval.py final_rerun.json          # determinism re-run
```

Total run time on 4 vCPU is about 35 minutes. `results/C_layer*.pt` (95 MB each) are gitignored; `s3_stats.py` recreates them.
