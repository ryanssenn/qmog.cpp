# Perplexity Investigation

Date: 2026-06-18

## Goal

Reduce perplexity and numerical drift in the int8 path, using the f32 path and Hugging Face goldens as reference signals.

## Initial Observations

- The repository already has component tests, logits/top-k comparison, and optional per-layer hidden-state stack comparison in `test/mistral`.
- README notes that the int8 path is faster but still has numerical drift and output quality issues.
- The default int8 export quantizes MLP `gate_proj` and `up_proj`, while `down_proj`, attention, embeddings, norms, and `lm_head` remain f32.
- The likely high-leverage area is the int8 matmul path used by MLP `gate_proj` and `up_proj`.
- Current worktree status was clean before implementation changes.

## Initial Hypotheses

1. Int8 drift may be dominated by quantization/dequantization mismatch in group scale application or group boundaries.
2. Int8 matmul accumulation order or SIMD behavior may differ enough from the scalar/f32 reference to amplify through the MLP.
3. Drift may first appear in MLP outputs rather than attention/KV cache if only MLP projections are quantized.
4. Existing layer-stack diagnostics should identify the first layer where hidden-state drift becomes significant.

## Log

### 2026-06-18 00:00 UTC - Repository orientation

- Listed repository files and searched for perplexity/logits/hidden/int8/quantization tooling.
- Found relevant files:
  - `src/backend/cpu/kernels.cpp`
  - `src/model/mistral/modules.cpp`
  - `src/loader/parameters.cpp`
  - `test/mistral/test_logits.cpp`
  - `test/mistral/test_cpu_kernels.cpp`
  - `scripts/test/mistral/logits.py`
- No implementation changes made yet.

## Baseline Measurements

### Environment / artifact availability

- `build/test_exec` and `build/mistral.cpp` already exist.
- `test/mistral/expected.txt` and `test/mistral/logits_expected.txt` exist.
- `mistral.bin` was missing from the repository root, so model-dependent tests could not run yet.
- Hugging Face checkout exists at `/home/ec2-user/Mistral-7B-v0.1`.
- Available disk space is about 49 GiB, sufficient for an int8 export.

Action taken: exported the default int8 binary with existing `export_mistral.py`.

### Baseline int8 test suite

Command: `./build/test_exec`

Artifact: `mistral.bin` is 17 GiB and reports `model: int8`.

Result: failed `test logits multi top10`; 6 / 7 tests passed.

Logits/top-k diagnostics:

| Prompt | Step | f32 top1 | int8 top1 | Status | top10 overlap | max aligned logit err |
|---|---:|---:|---:|---|---:|---:|
| paris | 0 | 4843 | 4843 | OK | 7/10 | 1.36799 |
| paris | 1 | 304 | 304 | OK | 5/10 | 1.74523 |
| paris | 2 | 272 | 272 | OK | 7/10 | 1.26449 |
| paris | 3 | 1080 | 1080 | OK | 7/10 | 0.907696 |
| paris | 4 | 1852 | 8376 | FLIP | 8/10 | 1.28044 |
| paris | 5 | 9504 | 9504 | OK | 3/10 | 5.45623 |
| sky | 0 | 264 | 5045 | FLIP | 6/10 | 0.770426 |
| sky | 1 | 4672 | 3534 | FLIP | 5/10 | 0 |
| sky | 2 | 5045 | 5045 | OK | 4/10 | 2.82281 |
| sky | 3 | 28723 | 302 | FLIP | 4/10 | 2.83138 |
| sky | 4 | 415 | 13 | FLIP | 6/10 | 3.60341 |
| sky | 5 | 4376 | 7212 | FLIP | 3/10 | 0 |

Notes:

- Current `test/mistral/logits_expected.txt` does not contain `layer_stack_*` keys, so `test layer stack prefill` is a skip-style pass and does not yet identify the first drift layer.
- Next action: regenerate existing logits goldens with `DUMP_LAYER_STACK=1` to enable the repository's layer-by-layer hidden-state diagnostic.

Perplexity baseline:

- Not available initially. Existing infrastructure had logits/top-k and hidden-state diagnostics but no prompt-level NLL/perplexity metric.
- After using the existing diagnostics, a small prompt-perplexity extension will be added to the same logits golden/test path, limited to the two existing short prompts.

### Layer-stack golden generation

Command: `DUMP_LAYER_STACK=1 venv/bin/python scripts/test/mistral/logits.py`

Result:

- Loaded `/home/ec2-user/Mistral-7B-v0.1` on CPU as `torch.bfloat16`.
- Rewrote `test/mistral/logits_expected.txt` with top-k logits plus `layer_stack_*` vectors.

### Baseline int8 test suite with layer-stack goldens

Command: `./build/test_exec`

Result: failed `test logits multi top10` and `test layer stack prefill`; 5 / 7 tests passed.

Layer-stack result:

- First reported hidden-state tolerance failure: prompt `paris`, layer 3, element 104.
- Expected `0.115234`, got `0.0584857`.

Updated logits/top-k diagnostics after regenerating HF goldens:

| Prompt | Step | f32 top1 | int8 top1 | Status | top10 overlap | max aligned logit err |
|---|---:|---:|---:|---|---:|---:|
| paris | 0 | 4843 | 4843 | OK | 7/10 | 1.43049 |
| paris | 1 | 304 | 304 | OK | 5/10 | 1.74523 |
| paris | 2 | 272 | 272 | OK | 7/10 | 1.26449 |
| paris | 3 | 1080 | 1080 | OK | 7/10 | 0.845196 |
| paris | 4 | 1852 | 8376 | FLIP | 8/10 | 1.34294 |
| paris | 5 | 9504 | 9504 | OK | 3/10 | 5.45623 |
| sky | 0 | 5045 | 5045 | OK | 6/10 | 0.770426 |
| sky | 1 | 28723 | 349 | FLIP | 5/10 | 2.93231 |
| sky | 2 | 13 | 13 | OK | 7/10 | 2.54843 |
| sky | 3 | 13 | 13 | OK | 4/10 | 4.90469 |
| sky | 4 | 1014 | 13 | FLIP | 2/10 | 0 |
| sky | 5 | 3181 | 3181 | OK | 5/10 | 0.541026 |

Next action: extend the existing layer-stack test with compact per-layer drift metrics (`max_abs`, `rmse`, cosine similarity) so drift growth can be measured instead of only reporting the first mismatched element.

## Code Changes

### Diagnostic change 1: layer-stack drift metrics

Files changed:

- `test/mistral/test_logits.cpp`

What changed:

- Added `max_abs`, `mean_abs`, `rmse`, cosine similarity, and first tolerance-breaking element index for layer-stack hidden-state comparisons.
- Kept the same failure condition: any element with absolute drift `>= 0.05` fails.

Why:

- The original layer-stack test stopped at the first mismatched element. That identified layer 3 but did not show whether drift was gradual, isolated, or explosive.

Result after rebuild and `./build/test_exec`:

- Logits test still failed with the same top1 flips.
- Layer-stack test now reports metrics.
- For prompt `paris`, drift was below tolerance through layer 2:
  - L0: `max_abs=0.00905616`, `rmse=0.00170572`
  - L1: `max_abs=0.0280727`, `rmse=0.00438504`
  - L2: `max_abs=0.0488212`, `rmse=0.0079189`
- First tolerance failure remains layer 3:
  - L3: `max_abs=0.0705551`, `rmse=0.0102978`, `first_bad=104`
- Drift grows gradually after that, then spikes at final decoder layer:
  - L30: `max_abs=1.61222`, `rmse=0.204356`
  - L31: `max_abs=140.202`, `rmse=5.48247`
  - final norm: `max_abs=15.7146`, `rmse=2.17268`

Interpretation:

- The first visible drift is not a KV-cache or tokenizer issue; it appears during decoder-layer accumulation with quantized MLP projections.
- The late layer spike suggests earlier MLP quantization error is amplified by later layers, and possibly that one or more final-layer activations are outlier-sensitive.

### Diagnostic change 2: run layer-stack metrics for all prompts

Files changed:

- `test/mistral/test_logits.cpp`

What changed:

- `test_layer_stack` now runs every prompt before returning failure, instead of stopping after the first failing prompt.

Why:

- Needed to confirm whether the first-drift layer is prompt-specific or common across the short eval set.

Result after rebuild and `./build/test_exec`:

- Logits test unchanged: failures at `paris` step 4 and `sky` steps 1 and 4.
- Layer-stack drift:
  - `paris`: first tolerance breach at layer 3 (`max_abs=0.0705551`, `rmse=0.0102978`).
  - `sky`: first tolerance breach at layer 2 (`max_abs=0.0529561`, `rmse=0.00569466`).
  - Both prompts show large layer-31 spikes:
    - `paris` L31: `max_abs=140.202`, `rmse=5.48247`
    - `sky` L31: `max_abs=143.962`, `rmse=6.24399`

Hypothesis:

- Since the int8 export quantizes only MLP `gate_proj` and `up_proj`, and drift starts after a few decoder layers then compounds, the first experiment should reduce weight quantization error in those projections.
- Most direct small experiment: reduce quantization group size from 64 to 32, re-export int8, and compare the same logits/layer-stack diagnostics.

### Code change 3: infer int8 group size from scale metadata

Files changed:

- `src/backend/cpu/kernels.cpp`

What changed:

- Removed hard-coded `GROUP_SIZE = 64` from int8 matmul.
- Derives `group_size` as `w.numel / w.scales.size()`.
- Added assertions that scale metadata is valid and row-aligned.

Why:

- This should be behavior-preserving for the current 64-group model while allowing controlled re-export experiments with different group sizes.

Result after rebuild and `./build/test_exec` with the existing 64-group `mistral.bin`:

- Behavior unchanged.
- Logits failures remain `paris` step 4 and `sky` steps 1 and 4.
- Layer-stack first tolerance failures remain:
  - `paris`: layer 3
  - `sky`: layer 2

Next action: reduce exporter `GROUP_SIZE` to 32, re-export int8, and rerun the same diagnostics.

### Experiment 1: group size 32

Files changed:

- `export_mistral.py`

What changed:

- Set exporter `GROUP_SIZE` from 64 to 32.
- Re-exported `mistral.bin`.

Why:

- Smaller quantization groups should reduce local weight reconstruction error for MLP `gate_proj` and `up_proj`.

Result after `./build/test_exec`:

- No improvement in greedy top1 flips:
  - `paris` step 4 still flips from 1852 to 8376.
  - `sky` steps 1 and 4 still flip.
- Some logit values improved locally (`sky` step 0 max aligned logit error dropped from `0.770426` to `0.263492`), but other steps regressed (`sky` step 3 max aligned logit error rose from `4.90469` to `6.93729`).
- Layer-stack drift regressed:
  - `paris` first tolerance breach moved earlier from layer 3 to layer 2.
  - `paris` final norm RMSE worsened from `2.17268` to `2.45763`.
  - `sky` final norm RMSE worsened from `2.25479` to `2.32447`.
- Runtime was also slower in this run (`test logits multi top10` ~71.9s vs ~8-9s cached baseline; some of this may be mmap/cache state, but more groups increase scale work).

Conclusion:

- Reducing all MLP quantization groups from 64 to 32 is a dead end for this short diagnostic set.
- Restore exporter default to 64 before trying a different hypothesis.

### Follow-up analysis: per-layer weight reconstruction error

Method:

- Used the same quantization/dequantization formula as `export_mistral.py` on HF MLP `gate_proj` and `up_proj` weights with group size 64.
- Computed per-layer relative RMSE, absolute RMSE, MAE, and max absolute weight error.

Result:

- Weight reconstruction error is very uniform across layers.
- Highest relative RMSE is around `0.00625`.
- Layer 31 is not a weight-error outlier:
  - L31 `gate_proj`: `rel_rmse=0.006033`, `rmse=0.00002167`, `max_abs=0.00022876`
  - L31 `up_proj`: `rel_rmse=0.006028`, `rmse=0.00001942`, `max_abs=0.00048828`

Conclusion:

- The layer-31 hidden-state spike is unlikely to be explained by unusually poor layer-31 weight quantization alone.

### Follow-up analysis: local quantized MLP output error on HF activations

Method:

- Loaded HF Mistral in `torch.bfloat16`.
- Registered hooks for each layer's `post_attention_layernorm` output and MLP output for the two test prompts.
- Recomputed each MLP locally using dequantized group-64 `gate_proj` and `up_proj`, f32 `down_proj`, and the captured HF MLP input.

Result:

- Local MLP output error is small on HF activations.
- `sky` top local MLP RMSE:
  - L31: `rmse=0.001294`, `max_abs=0.044574`, `rel_rmse=0.003734`
  - L30: `rmse=0.000886`, `max_abs=0.008977`, `rel_rmse=0.005872`
- `paris` top local MLP RMSE:
  - L31: `rmse=0.001034`, `max_abs=0.019228`, `rel_rmse=0.003741`
  - L30: `rmse=0.000791`, `max_abs=0.006093`, `rel_rmse=0.005713`

Conclusion:

- On the reference trajectory, group-64 MLP quantization introduces small local MLP output error.
- The large late-layer C++ hidden-state drift is more likely accumulated earlier drift interacting with later layer sensitivity than a single bad quantized weight tensor.

Next action:

- Re-export f32 and run the full f32 test suite as a reference sanity check against the regenerated HF layer-stack/logits goldens.

### F32 reference sanity check attempt

Action:

- Exported `mistral.bin` with `--quant f32`.
- Started `./build/test_exec`.

Result:

- Artifact size: 27 GiB.
- Early kernel tests passed (`rope`, `matmul`, `row matmul`, `softmax`, `silu`).
- Stopped the run before completion because full f32 model diagnostics are too slow for the current int8-focused iteration.

Conclusion:

- Do not use full f32 suite runs during iteration.
- Future f32 checks should be limited to short prompts, component-level checks, or targeted Python/HF probes.
- Next action: re-export int8 with restored group size 64 and continue int8-only diagnostics.

### Restored default int8 artifact

Action:

- Restored `export_mistral.py` to `GROUP_SIZE = 64`.
- Re-exported default int8 `mistral.bin`.
- Reran `./build/test_exec`.

Result:

- Artifact size: 17 GiB.
- Diagnostics match the original group-64 baseline:
  - Logits failures: `paris` step 4, `sky` steps 1 and 4.
  - First layer-stack tolerance failures: `paris` layer 3, `sky` layer 2.
  - Layer-31 spikes remain: `paris rmse=5.48247`, `sky rmse=6.24399`.

Next action:

- Use a short HF/Python probe to test selective precision before changing C++ runtime structure for mixed f32/int8 layers.

### Prompt perplexity diagnostic

Files changed:

- `scripts/test/mistral/logits.py`
- `test/mistral/test_logits.cpp`
- `test/mistral/main.cpp`

What changed:

- Extended the existing logits golden path to dump per-token NLL, mean NLL, and PPL for the two existing short prompts.
- Added a diagnostic C++ test that scores the same prompts with the active model.
- Adjusted the test runner to show output for `test prompt perplexity` even when it passes.

Result on default int8 group-64:

- `paris`: reference mean NLL `3.28117`, int8 mean NLL `3.04078`; reference PPL `26.6069`, int8 PPL `20.9215`.
- `sky`: reference mean NLL `3.27475`, int8 mean NLL `3.18235`; reference PPL `26.4367`, int8 PPL `24.1034`.

Interpretation:

- On this tiny prompt-level metric, int8 is lower-NLL than the HF reference, even while top-k/logit distribution drift is clearly worse.
- This short PPL metric is useful as a guardrail but is too small and biased to optimize alone.

### Diagnostic change 4: hidden-state RMS ratio

Files changed:

- `test/mistral/test_logits.cpp`

What changed:

- Added `rms_ratio = got_rms / reference_rms` to layer-stack drift output.

Result on default int8 group-64:

- The int8 hidden state usually has lower RMS than the HF reference through early and mid layers.
- Examples:
  - `paris` L2 `rms_ratio=0.675697`; L3 `0.841979`; L10 `0.968258`.
  - `sky` L3 `rms_ratio=0.820866`; L10 `0.903008`; L20 `0.923952`.
- Final layer pre-norm ratio is about `0.10`, but final norm brings RMS back closer (`paris norm=0.979747`, `sky norm=0.938056`).

Hypothesis:

- Quantized MLP updates may be underpowered relative to the reference trajectory. A small int8-only MLP output gain might reduce hidden drift and top-k/logit error.

Next experiment:

- Apply an int8-only MLP output gain of `1.05` after `down_proj` and before the residual add.

### Experiment 3: int8-only MLP output gain 1.05

Files changed:

- `src/model/mistral/modules.cpp`

What changed:

- Added `mul(infer.hidden_state, infer.hidden_state, 1.05f)` after int8 MLP `down_proj`.

Why:

- RMS ratios suggested int8 hidden states are often lower-norm than the HF reference through early and mid layers.

Result:

- Top1 flips did not improve:
  - `paris` step 4 still flipped.
  - `sky` steps 1 and 4 still flipped.
- Prompt PPL regressed relative to baseline:
  - `paris` int8 PPL `20.9215 -> 21.0586`
  - `sky` int8 PPL `24.1034 -> 24.7252`
- Hidden drift did not materially improve; first failing layers stayed the same.
- Some RMS ratios moved closer to 1, but this did not translate to better logits or PPL.

Conclusion:

- A simple global int8 MLP output gain is not a useful fix.
- Revert the gain.

### Root-cause candidate: exporter quantizes from bfloat16

Observation:

- HF safetensors for Mistral are `torch.bfloat16`.
- `export_mistral.py::quantize` was using the tensor's existing dtype for scale computation and rounding.

Measurement on representative tensors:

| Tensor | Quantization math | RMSE | MAE | Max abs reconstruction error |
|---|---:|---:|---:|---:|
| L0 gate_proj | current bf16 | 0.000642328 | 0.0000413167 | 0.199609 |
| L0 gate_proj | float32 | 0.0000197512 | 0.0000165088 | 0.000461909 |
| L31 up_proj | current bf16 | 0.000607008 | 0.0000393727 | 0.120909 |
| L31 up_proj | float32 | 0.0000194221 | 0.0000163584 | 0.000488281 |

Interpretation:

- This is a much stronger error source than group size or accumulation order.
- Quantization should cast source weights to float32 before computing scales and rounded int8 values.

Next experiment:

- Patch `quantize` to use `x.to(torch.float32)` before grouping/scaling/rounding.
- Re-export default int8 and rerun diagnostics.

### Code change 5: quantize weights from float32

Files changed:

- `export_mistral.py`

What changed:

- `quantize` now casts source tensors to `torch.float32` before reshaping, scale computation, and rounding.

Why:

- The Mistral safetensors are `torch.bfloat16`.
- Quantizing directly from bf16 introduced large reconstruction errors.

Result after re-exporting default int8 group-64 and running `./build/test_exec`:

- `paris` step 4 top1 flip was fixed:
  - Before: f32 `1852`, int8 `8376`.
  - After: f32 `1852`, int8 `1852`.
- Remaining top1 flips:
  - `sky` step 0: f32 `5045`, int8 `264` (top displayed values are close/tied; top10 overlap 7/10).
  - `sky` step 1: f32 `28723`, int8 `28725`.
  - `sky` step 4: f32 `1014`, int8 `13`.
- Prompt PPL moved much closer to reference:
  - `paris`: int8 PPL `20.9215 -> 25.7102`, reference `26.6069`; mean NLL delta `-0.240395 -> -0.0342839`.
  - `sky`: int8 PPL `24.1034 -> 25.2733`, reference `26.4367`; mean NLL delta `-0.0924022 -> -0.0450063`.
- Layer drift:
  - First tolerance failures remain early (`paris` L3, `sky` L2).
  - Final norm RMSE improved:
    - `paris`: `2.17268 -> 2.06641`
    - `sky`: `2.25479 -> 1.97992`

Conclusion:

- This is the first clear quality improvement.
- The dominant source found so far was bf16 arithmetic during export-time quantization.

Next experiment:

- Retest group size 32 now that quantization math is fixed. The previous group-size-32 regression was confounded by bf16 quantization.

### Experiment 4: group size 32 after float32 quantization fix

Files changed:

- `export_mistral.py`

What changed:

- Temporarily changed `GROUP_SIZE` from 64 to 32 after fixing quantization math to float32.
- Re-exported int8 and reran diagnostics.

Result:

- Quality was effectively neutral compared with fixed group-64:
  - `paris` PPL `25.7102 -> 25.6541`? actually mean NLL moved `3.24689 -> 3.24471` in the group-32 run relative to group-64, a tiny mixed change; both remain near reference.
  - `sky` PPL `25.2733 -> 25.3119`, also tiny.
  - Same remaining top1 flips on `sky` steps 0, 1, and 4.
  - Layer-stack metrics nearly identical.
- Runtime regressed heavily in this run:
  - `test logits multi top10` about 99.7s versus much shorter cached group-64 runs.

Conclusion:

- Group size 32 is not worth keeping for the current implementation and short diagnostic set.
- Restore group size 64.

### HF probe: dequantized MLP weights, teacher forced

Method:

- Loaded HF model in `torch.bfloat16`.
- Replaced MLP `gate_proj` and `up_proj` weights with group-64 quantize/dequantize values using the same scale convention as the exporter.
- Ran the same two short prompts and six teacher-forced steps as `test_logits.cpp`.
- Also tried keeping the final layer f32.

Important correction:

- An earlier version of this probe used each experiment's own greedy continuation. That conflated quantization error with trajectory drift, so those results were discarded.

Results:

- All MLP `gate_proj`/`up_proj` dequantized:
  - 1 top1 flip over 12 prompt/step checks.
  - `sky`: 0 flips, top10 overlap 10/10 at every step.
  - `paris`: only step 2 flipped (`272` to `624`), top10 overlap still 10/10.
- Keeping final layer f32:
  - 2 top1 flips over 12 checks.
  - `sky` step 0 flipped (`5045` to `264`) despite 10/10 top10 overlap.
  - `paris` step 2 still flipped (`272` to `624`).

Interpretation:

- Pure HF dequantized MLP weights are much closer to HF reference than the C++ int8 run on the same short teacher-forced diagnostics.
- Selectively keeping only the final layer f32 is not supported by this probe.
- Remaining suspects are C++ runtime differences under token-by-token KV-cache execution, int8 matmul arithmetic details, or accumulated activation drift that the full-sequence HF probe does not reproduce.

### Config check

- `../Mistral-7B-v0.1/config.json` has `rms_norm_eps: 1e-05`.
- Current `RMSNorm` hard-coded epsilon is also `1e-5f`.
- No epsilon mismatch found.

### HF probe: cache execution

Method:

- Ran HF model token-by-token with `past_key_values`, matching C++ KV-cache style.
- Compared against the full-sequence HF goldens.
- Repeated with group-64 dequantized MLP `gate_proj`/`up_proj`.

Results:

- HF bfloat16 cache vs full-sequence reference:
  - 1 top1 flip over 12 checks: `paris` step 2 (`272` to `624`), with 10/10 top10 overlap.
- HF cache plus dequantized MLP weights:
  - 2 top1 flips over 12 checks:
    - `sky` step 0 (`5045` to `264`), with 10/10 top10 overlap and equal displayed top logit (`12.0`).
    - `paris` step 2 (`272` to `624`), with 10/10 top10 overlap.

Interpretation:

- Some top1 failures are tie/order sensitivity rather than large distribution drift.
- HF cache plus dequantized MLP weights still remains much closer than the current C++ int8 diagnostics, so C++ int8 runtime arithmetic remains a likely contributor.

Next experiment:

- Change C++ int8 matmul accumulation to dequantize each element before multiply/sum, instead of accumulating a raw int8 dot per group and scaling the group sum afterward. This tests whether group-level scale application/accumulation order is a dominant source.

### Experiment 2: per-element dequant accumulation in int8 matmul

Files changed:

- `src/backend/cpu/kernels.cpp`

What changed:

- In int8 matmul, replaced `sum += inv_scale * dot_i8_f32(...)` with an inner loop:
  - dequantize each int8 weight with `inv_scale`,
  - multiply by activation,
  - accumulate in row order.

Why:

- Tests whether applying scale after a grouped int8 dot product causes meaningful numerical drift.

Result after rebuild and `./build/test_exec`:

- Logits diagnostics were effectively unchanged.
- Layer-stack metrics were effectively unchanged.
- Runtime regressed heavily:
  - `test logits multi top10` increased to about 85.6s.

Conclusion:

- Group-level scale application / dot accumulation order is not the dominant source of the observed drift.
- Revert to the grouped dot implementation for performance.

### HF probe: hidden-state cache vs full-sequence comparison

Method:

- Loaded HF model in `torch.bfloat16`.
- Compared hidden states for the last prompt token between:
  - full-sequence forward with `use_cache=False`,
  - token-by-token forward with `past_key_values`.

Results:

- `sky`:
  - Early layers are nearly identical through layer 5 (`L05 rmse=0.000157`).
  - First `max_abs >= 0.05` appears late at layer 23.
  - L31 / final norm: `max_abs=1.5`, `rmse=0.059007`.
- `paris`:
  - Early layers are nearly identical through layer 5 (`L05 rmse=0.000175`).
  - First `max_abs >= 0.05` appears at layer 31.
  - L31 / final norm: `max_abs=0.5`, `rmse=0.058417`.

Interpretation:

- Comparing C++ token-by-token hidden states against HF full-sequence hidden states does add some late-layer noise.
- It does not explain the C++ int8 first drift at layer 2-3.
- It also does not explain the much larger C++ int8 final-layer RMSE (`5.48` to `6.24`), though it means the absolute layer-31 spike is partly inflated by cache-vs-full comparison.

## Dead Ends / Regressions

- Group size 32 before fixing quantization math: regressed early hidden drift and did not reduce top1 flips.
- Full f32 test-suite run: stopped because it is too slow for iteration; future f32 checks should be short and targeted.
- Keeping only the final layer f32 in a HF qdq probe: did not help.
- Per-element dequant accumulation in C++ int8 matmul: produced effectively identical quality and was much slower.
- Global int8 MLP output gain `1.05`: did not reduce flips or drift and worsened short prompt PPL.
- Group size 32 after fixing quantization math: essentially neutral quality and slower, so group size 64 remains preferred.

## Final Conclusions

### What improved

- The main improvement came from fixing export-time quantization to compute scales and rounded int8 weights in float32 instead of bfloat16.
- Short prompt PPL moved closer to HF reference:
  - `paris`: int8 PPL `20.9215 -> 25.7102`, reference `26.6069`.
  - `sky`: int8 PPL `24.1034 -> 25.2733`, reference `26.4367`.
- Teacher-forced logits improved:
  - `paris` step 4 top1 flip was fixed (`8376 -> 1852`).
- Final norm hidden-state RMSE improved:
  - `paris`: `2.17268 -> 2.06641`
  - `sky`: `2.25479 -> 1.97992`

### What remains

- The int8 path still fails the top-k logits test due to `sky` top1 flips at steps 0, 1, and 4.
- Layer-stack drift still first exceeds tolerance early:
  - `paris`: layer 3.
  - `sky`: layer 2.
- Some top1 failures are near-tie/order sensitive, but distribution drift remains real, especially in later `sky` steps.

### Current selected configuration

- Exporter group size: 64.
- Quantized tensors: MLP `gate_proj` and `up_proj`.
- Quantization math: float32.
- Runtime int8 matmul: grouped int8 dot with scale applied per group, group size inferred from scale metadata.

### Next steps

- Add cache-based HF layer-stack goldens or a separate cache-aware diagnostic so late-layer hidden drift is not inflated by cache-vs-full comparison.
- Investigate the remaining early drift by instrumenting one layer at a time around attention vs MLP outputs on the actual token-by-token path.
- Consider mixed precision for the earliest MLP layers only if layer-local diagnostics show a clear payoff; avoid a broad mixed-precision refactor until that evidence exists.
