# Attention Extraction Pipeline — Fixed & Extended

## What changed vs the originals

### `extract_attention.py`

| Issue in original | Fix |
|---|---|
| Only ran CoT prompts | Now runs **both ZS and CoT** per scenario. Output `.npz` has an extra `prompt_type` axis. |
| `model.generate(output_attentions=True)` stored attention at every gen step → **OOM** | **Decoupled**: forward pass for attention (one-shot, cheap), separate `generate()` for decision text (no attention storage). ~10× less VRAM. |
| `torch.float16` | `torch.bfloat16` — matches main experiment (L40S/A100 both support it). |
| `max_new_tokens=200` truncated CoT JSON | ZS=120, CoT=350. |
| No Qwen `<think>` handling | Stripped with regex before JSON parsing. |
| Silent if a section marker missing in prompt | Prints `! section 'X' not found`. |
| 8 scenarios | 10 scenarios. Added 2 "Ministral-traps" (6, 10) where the doctrinal answer is unambiguous and the empathy pull is mild — if Ministral-ZS still picks citizens, the failure is clearly attention-routing, not task-ambiguity. |

### `visualize_attention.py`

| Issue in original | Fix |
|---|---|
| `axhline(transform=ax.transAxes)` crashed in matplotlib ≥3.8 | Replaced with `ax.plot([0,1], [y,y], transform=ax.transAxes)`. |
| Couldn't show ZS vs CoT | Decision matrix now has both columns per model. Doctrine-survival plot overlays ZS (dashed) vs CoT (solid) per model. |
| No focused failure-mode view | New **Fig 6: `attention_ministral_failure`** — two-panel figure showing (a) Ministral's doctrine attention by layer ZS-vs-CoT, (b) per-scenario decisions ZS-vs-CoT. This is the headline figure for the architecture-matters argument. |
| Figures saved to working directory | All outputs go to `attention_figures/`. |

### `run_attention_extraction.slurm`

- Walltime raised from 2h → 3h (now 2× the work due to ZS+CoT).
- Logs `nvidia-smi --query-gpu=name,compute_cap` so you have a record of which GPU was assigned.

## How to run

```bash
# On the cluster:
sbatch run_attention_extraction.slurm

# Once attention_data/*.npz exist (transfer them down or run on cluster head):
python visualize_attention.py
```

## What the figures will tell you

1. `attention_decisions.pdf` — At-a-glance: ZS vs CoT decisions per scenario per model. Ministral-ZS row should be mostly "CITIZENS"; Ministral-CoT should show pillar picks recovering, especially on scenarios 6, 9, 10.

2. `attention_doctrine_survival.pdf` — One subplot, 6 lines: each model × {ZS dashed, CoT solid}. Ministral-ZS line drops sharply with depth; Ministral-CoT stays elevated. Llama and Qwen3 lines should be relatively flat by comparison.

3. `attention_section_heatmap.pdf` — Per-layer attention distribution across sections, CoT. Shows what each model "looks at."

4. `attention_doctrine_vs_context.pdf` — Per-scenario ratio of doctrine attention vs context attention. Useful for showing scenario-specific behavior.

5. `attention_difference_map.pdf` — Where Qwen attends more than Llama / more than Ministral. Should show green (Qwen higher) on the `Doctrine` row, especially in mid-to-deep layers.

6. **`attention_ministral_failure.pdf`** — The headline figure. Left: Ministral's doctrine attention collapses in ZS, recovers under CoT. Right: ZS picks citizens for almost every scenario; CoT recovers pillar picks specifically on the obvious-pillar scenarios (6, 9, 10).

## If something still fails

- **`AssertionError` or shape mismatch in attention** → almost always due to a tokenizer config issue. Print `inputs["input_ids"].shape` and `attentions[0].shape` to debug.
- **OOM despite the fix** → reduce `MAX_NEW_TOKENS["cot"]` to 250, or run one model at a time by commenting others out of `MODELS`.
- **All decisions "parse_error"** → check `raw_text` in the saved decisions JSON; model may be wrapping JSON in markdown fences. Add `re.sub(r"```(?:json)?", "", text)` in `parse_decision`.
- **Section markers not found** → the chat template may be wrapping your `<world>` etc. in additional tokens. Inspect `full_text` once and verify the markers appear verbatim.

## Output `.npz` schema (for downstream analysis)

```
section_attention      [n_prompt_types=2, n_scenarios=10, n_layers, n_sections=8]
layer_world_attention  [n_prompt_types=2, n_scenarios=10, n_layers]
head_world_vs_context  [n_prompt_types=2, n_scenarios=10, n_layers, n_heads, 2]
                                                                          # [..,0]=world, [..,1]=context
section_names          ["role","persona","world","scenario",
                        "memory","context","task","output"]
prompt_types           ["zeroshot","cot"]
scenario_labels        ["1_doctrine_vs_empathy", ..., "10_lone_pillar"]
decisions              JSON list of length 20 (10 scenarios × 2 prompt types)
                         each entry: prompt_type, scenario_idx, scenario_label,
                         target, attribute, pillar_institution, method,
                         observation, strategic_intent, raw_text
```
