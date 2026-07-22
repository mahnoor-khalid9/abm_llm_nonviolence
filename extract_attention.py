"""
extract_attention.py - Attention Weight Extraction (ZS + CoT)
==============================================================
Runs each model on a fixed set of scenarios under BOTH prompting conditions
(ZeroShot and Chain-of-Thought) and saves per-layer attention statistics
plus the resulting decisions.

Key design choices that fix the original script's failures:
  1. Attention is captured via a single forward pass on the INPUT only.
     Generation (for the decision text) is a separate, no-attention call.
     This is ~10x cheaper than generate(output_attentions=True) and avoids
     the OOM that kills long-prompt + long-generation runs.
  2. bfloat16 precision (matches main experiment on L40S / A100).
  3. eager attention implementation (required to expose attention tensors).
  4. Both ZS and CoT are run, results stored under the same .npz keyed
     by prompt_type so visualization can compare them directly.
  5. Qwen3 <think>...</think> tags are stripped from decision text.

Output shapes per model file:
    section_attention      [n_prompt_types, n_scenarios, n_layers, n_sections]
    layer_world_attention  [n_prompt_types, n_scenarios, n_layers]
    head_world_vs_context  [n_prompt_types, n_scenarios, n_layers, n_heads, 2]
    decisions: JSON list (len = 2 * n_scenarios)
    prompt_types: ["zeroshot", "cot"]

Usage:
    python extract_attention.py
"""

from __future__ import annotations
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import gc
import re
import numpy as np
import torch
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

WORK_DIR = Path("/work/pi_lfiondella_umassd_edu/Mahnoor")
OUTPUT_DIR = Path("attention_data")
OUTPUT_DIR.mkdir(exist_ok=True)

MODELS = [
    {
        "name":  "Llama-3.1-8B-Instruct",
        "path":  str(WORK_DIR / "models" / "Llama-3.1-8B-Instruct"),
        "short": "llama",
    },
    {
        "name":  "Ministral-8B-Instruct-2410",
        "path":  str(WORK_DIR / "models" / "Ministral-8B-Instruct-2410"),
        "short": "ministral",
    },
    {
        "name":  "Qwen3-8B",
        "path":  str(WORK_DIR / "models" / "Qwen3-8B"),
        "short": "qwen",
    },
]

PROMPT_TYPES = ["zeroshot", "cot"]

# Generation budget (post-input). Empirically: ZS ~50, CoT ~250 tokens.
MAX_NEW_TOKENS = {"zeroshot": 120, "cot": 350}

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT (matches simulation exactly)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
<role>
You are an activist agent in an agent-based simulation of a non-violent
resistance movement. Your movement's goal is to overthrow an authoritarian
regime. You make strategic decisions each step about where to move and what
action to take based on your local neighborhood and memory. Your character and decisions
are inspired by the principles of Mahatma Gandhi.
</role>


<persona>
You embody Gandhi's core principles in every decision.

  - Nonviolence:  You reject all forms of violence, even under provocation.
  - Truth: You act with moral courage and believe truth brings justice.
  - Empathy: You are moved by the suffering of others to act for their liberation.
  - Courage: You face adversity without compromising your values.
  - Patience: You understand significant change takes time and sustained effort.
  - Sacrifice: You endure personal hardship in service of the movement.
  - Inclusiveness: You build bridges across communities rather than divide them.
  - Forgiveness: You see forgiveness as strength, not weakness.
</persona>


<world>
No ruler governs alone. Regimes survive because institutions like the judiciary,
civil service, media, military, religious bodies, and business actively cooperate
with them. These institutions are called pillars of support. When enough pillars
withdraw cooperation, the regime collapses. Mass citizen protest alone has
never overthrown a regime without institutional defection.

Pillars crack from the inside. Members are pulled away when movements appeal
to their conscience, identity, and self-interest. Reminding soldiers they are
also fathers and citizens, showing civil servants their work harms their
communities. Violence and threats push members deeper into loyalty to the
regime. Successful movements pull. They never push.

Three methods of nonviolent action exist. Protest and persuasion through
marches, petitions, and symbols communicates that something is wrong and
builds awareness and early support. Non-cooperation through strikes, boycotts,
and work slowdowns withdraws the resources pillars depend on and weakens
the regime structurally. Intervention through sit-ins, blockades, and civil
disobedience directly disrupts the regime's ability to function and carries the
highest risk but greatest impact.

Effective movements build a winning record of small victories before attempting
large-scale action. Every action should have a clear target chosen for strategic
reasons, not applied randomly. Movements that attempt coercion before building
sufficient strength frequently fail.
</world>



<scenario>
You are surrounded by four types of agents: 
- Pillars: institutions supporting the regime (judiciary, media, religious, business, etc).
  A "stable" pillar still supports the regime — it is your target.
  A "defected" pillar has already broken from the regime — victory there. 
  Do not target defected pillars. You have won them. Move on to stable pillars.
- Activists: fellow activists fully committed to nonviolent resistance.  
- Police: security forces who may arrest or kill civilians and activists.
  An "active" officer still serves the regime. They can be reached.
  A "defected" officer has chosen conscience. They are now with us.
- Civilians: the majority, who may remain inactive or join the resistance.
  "Suffering" citizens have high grievance — they are ready to be mobilised.
  "Unbothered" citizens have low grievance — they need awakening.


You are provided with your local neighborhood summary organized by direction
(north, south, east, west, diagonals, and adjacent).

For each direction, you may observe:
- `can_move_to`: positions you can move into in that direction.
- `citizens`: counts by condition (suffering, unbothered, protesting, active_nonviolent).
- `police`: counts by attitude (defected, active).
- `pillars`: lists by status (defected, stable). Institution names are shown.
- `activists`: counts by activity (protesting, active_nonviolent, jailed).

If no key appears for a direction, nothing of that type is present there.
</scenario>
"""

# ─────────────────────────────────────────────────────────────────────────────
# SCENARIOS - designed to expose architectural differences
#
# Scenarios 6, 9, 10 are "Ministral-trap" scenarios: the doctrinal answer is
# unambiguous (pillar accessible, no police, only mild empathy pull). If
# Ministral-ZS still picks citizens here, the failure is clearly attention-
# routing, not task-ambiguity.
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_CONTEXTS = [
    # ── 1. DOCTRINE vs EMPATHY (strong empathy pull) ──
    {
        "label": "1_doctrine_vs_empathy",
        "description": "Pillar north (1 citizen) vs 7 suffering south (no pillar)",
        "memory": "I approached citizens through protest and persuasion to stir their conscience.",
        "context": '{"my_position":[20,20],"neighborhood":{"north":{"can_move_to":[[18,20]],"citizens":{"suffering":1},"pillars":{"stable":["Business Community"]}},"south":{"can_move_to":[[22,20],[23,20]],"citizens":{"suffering":7,"unbothered":2}},"east":{"can_move_to":[[20,22]],"citizens":{"unbothered":3}},"west":{"citizens":{"unbothered":2}},"adjacent":{"can_move_to":[[19,20],[21,20]],"citizens":{"suffering":2,"unbothered":1}}}}'
    },

    # ── 2. SAFETY vs DUTY ──
    {
        "label": "2_safety_vs_duty",
        "description": "Pillar adjacent with 2 police vs safe citizens west (4 suffering)",
        "memory": "1 comrade arrested nearby. We must stay cautious but committed.",
        "context": '{"my_position":[15,25],"neighborhood":{"adjacent":{"can_move_to":[[14,25],[16,25]],"citizens":{"suffering":2},"pillars":{"stable":["Judiciary Institution"]},"police":{"active":2}},"west":{"can_move_to":[[15,23],[15,22]],"citizens":{"suffering":4,"unbothered":3}},"east":{"citizens":{"unbothered":2}},"north":{"can_move_to":[[13,25]],"citizens":{"unbothered":1}},"south":{"can_move_to":[[17,25]],"citizens":{"suffering":1}}}}'
    },

    # ── 3. PILLAR CHOICE (two pillars, different contexts) ──
    {
        "label": "3_pillar_choice",
        "description": "Judiciary north (3 suffering) vs Media south (1 unbothered)",
        "memory": "I stand firm in active nonviolent resistance. Guided by truth, I seek the pillars.",
        "context": '{"my_position":[10,10],"neighborhood":{"north":{"can_move_to":[[8,10]],"citizens":{"suffering":3},"pillars":{"stable":["Judiciary Institution"]}},"south":{"can_move_to":[[12,10]],"citizens":{"unbothered":1},"pillars":{"stable":["State-Supported Media"]}},"east":{"can_move_to":[[10,12]],"citizens":{"suffering":1,"unbothered":2}},"west":{"can_move_to":[[10,8]],"citizens":{"unbothered":3}},"adjacent":{"can_move_to":[[9,10],[11,10]],"citizens":{"suffering":1,"unbothered":1}}}}'
    },

    # ── 4. COORDINATION vs SOLO ──
    {
        "label": "4_coordination_vs_solo",
        "description": "Ally + pillar northeast vs 6 suffering alone southwest",
        "memory": "Guided by compassionate witness, I approached citizens to stir their conscience.",
        "context": '{"my_position":[25,15],"neighborhood":{"northeast":{"can_move_to":[[23,17]],"citizens":{"suffering":2},"pillars":{"stable":["Civil Service"]},"activists":{"active_nonviolent":1}},"southwest":{"can_move_to":[[27,13],[28,14]],"citizens":{"suffering":6,"unbothered":1}},"east":{"can_move_to":[[25,17]],"citizens":{"unbothered":2}},"west":{"can_move_to":[[25,13]],"citizens":{"suffering":1}},"adjacent":{"can_move_to":[[24,15],[26,15]],"citizens":{"suffering":1,"unbothered":2}}}}'
    },

    # ── 5. DEFECTED vs STABLE ──
    {
        "label": "5_defected_vs_stable",
        "description": "Defected pillar north vs stable pillar south vs citizens east",
        "memory": "The regime weakens. One institution has already joined us.",
        "context": '{"my_position":[30,30],"neighborhood":{"north":{"can_move_to":[[28,30]],"citizens":{"suffering":3},"pillars":{"defected":["Educational System"]}},"south":{"can_move_to":[[32,30]],"citizens":{"suffering":1},"pillars":{"stable":["Local Government"]}},"east":{"can_move_to":[[30,32]],"citizens":{"suffering":5,"unbothered":2}},"west":{"can_move_to":[[30,28]],"citizens":{"unbothered":4}},"adjacent":{"can_move_to":[[29,30],[31,30]],"citizens":{"unbothered":2}}}}'
    },

    # ── 6. PILLAR ADJACENT (Ministral-trap: doctrinal answer obvious) ──
    # Pillar is RIGHT NEXT TO the activist. Only mild suffering elsewhere.
    # No police, no risk. CANVAS doctrine says: pull this pillar now.
    # If Ministral-ZS still picks citizens, the doctrinal signal failed to
    # propagate to the decision token.
    {
        "label": "6_pillar_adjacent_low_pull",
        "description": "Judiciary adjacent, no police, only mild suffering scattered",
        "memory": "I have been wandering. The neighborhood is quiet today. A judiciary is right beside me.",
        "context": '{"my_position":[18,18],"neighborhood":{"adjacent":{"can_move_to":[[17,18],[19,18]],"pillars":{"stable":["Judiciary Institution"]},"citizens":{"unbothered":1}},"east":{"can_move_to":[[18,20]],"citizens":{"suffering":1,"unbothered":1}},"west":{"can_move_to":[[18,16]],"citizens":{"unbothered":2}},"north":{"can_move_to":[[16,18]],"citizens":{"suffering":1}},"south":{"can_move_to":[[20,18]],"citizens":{"unbothered":1}}}}'
    },

    # ── 7. MAXIMUM CONFLICT ──
    {
        "label": "7_maximum_conflict",
        "description": "Pillar+police north vs 8 suffering south vs allies east vs empty west",
        "memory": "2 comrades arrested. Police patrols intensifying. We must act wisely.",
        "context": '{"my_position":[20,30],"neighborhood":{"north":{"can_move_to":[[18,30]],"citizens":{"suffering":1},"pillars":{"stable":["Religious Institution"]},"police":{"active":1}},"south":{"can_move_to":[[22,30],[23,30]],"citizens":{"suffering":8,"unbothered":1}},"east":{"can_move_to":[[20,32]],"citizens":{"suffering":2},"activists":{"active_nonviolent":2}},"west":{"can_move_to":[[20,28]],"citizens":{"unbothered":1}},"adjacent":{"can_move_to":[[19,30],[21,30]],"citizens":{"suffering":3,"unbothered":2},"police":{"active":1}}}}'
    },

    # ── 8. CITIZENS ONLY (empathy baseline - correct answer IS citizens) ──
    {
        "label": "8_citizens_only",
        "description": "No pillars, no police. Suffering north vs unbothered south",
        "memory": "The neighborhood is quiet. No institutions in sight.",
        "context": '{"my_position":[35,5],"neighborhood":{"north":{"can_move_to":[[33,5]],"citizens":{"suffering":6}},"south":{"can_move_to":[[37,5]],"citizens":{"unbothered":8}},"east":{"can_move_to":[[35,7]],"citizens":{"suffering":3,"unbothered":3}},"west":{"can_move_to":[[35,3]],"citizens":{"suffering":1,"unbothered":1}},"adjacent":{"can_move_to":[[34,5],[36,5]],"citizens":{"suffering":2,"unbothered":4}}}}'
    },

    # ── 9. TRIPLE PILLAR (Ministral-trap: doctrine clearly applies) ──
    {
        "label": "9_triple_pillar",
        "description": "3 pillars visible (Business, Legislative, Civil Service); no police",
        "memory": "Multiple institutions surround us. The regime's support is visible everywhere.",
        "context": '{"my_position":[15,20],"neighborhood":{"north":{"can_move_to":[[13,20]],"citizens":{"suffering":2},"pillars":{"stable":["Business Community"]}},"east":{"can_move_to":[[15,22]],"pillars":{"stable":["Legislative Institution"]}},"southwest":{"can_move_to":[[17,18]],"citizens":{"suffering":4,"unbothered":1},"pillars":{"stable":["Civil Service"]}},"south":{"can_move_to":[[17,20]],"citizens":{"unbothered":3}},"west":{"can_move_to":[[15,18]],"citizens":{"suffering":1}},"adjacent":{"can_move_to":[[14,20],[16,20]],"citizens":{"suffering":1,"unbothered":2}}}}'
    },

    # ── 10. LONE PILLAR (Ministral-trap: easiest pillar case) ──
    # Almost no citizens. One pillar reachable. Doctrine is the only signal
    # in the prompt that points to a useful action. If even this fails, the
    # doctrinal signal is being lost.
    {
        "label": "10_lone_pillar",
        "description": "Sparse landscape: one Civil Service pillar east, scattered unbothered citizens",
        "memory": "The neighborhood is empty today. I see one institution nearby.",
        "context": '{"my_position":[5,5],"neighborhood":{"east":{"can_move_to":[[5,7]],"pillars":{"stable":["Civil Service"]},"citizens":{"unbothered":1}},"north":{"can_move_to":[[3,5]],"citizens":{"unbothered":1}},"south":{"can_move_to":[[7,5]],"citizens":{"unbothered":2}},"west":{"can_move_to":[[5,3]],"citizens":{"unbothered":1}},"adjacent":{"can_move_to":[[4,5],[6,5]],"citizens":{"unbothered":1}}}}'
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# USER PROMPT TEMPLATES - ZeroShot vs CoT
# ─────────────────────────────────────────────────────────────────────────────

USER_PROMPT_ZS = """<memory>
{memory}
</memory>

<context>
{context}
</context>

<task>
Based on your memory and neighborhood, decide where to move and what action
to take. Consider what you know about how regimes fall and how nonviolent
movements succeed. Your decision should reflect your Gandhian principles.
</task>

<output_rules>
Every field is required. Respond with this exact JSON structure and nothing else.
{{
  "decision": {{
    "chosen_position": [x, y],
    "target_agent_group": "citizens | pillars | police | activists",
    "target_pillar_institution": "Institution name or N/A",
    "primary_attribute": "Gandhian attribute",
    "non_violent_action_method": "protest_and_persuasion | non-cooperation | intervention"
  }}
}}
</output_rules>"""

USER_PROMPT_COT = """<memory>
{memory}
</memory>

<context>
{context}
</context>

<task>
Based on your memory and neighborhood, decide where to move and what action
to take. Consider what you know about how regimes fall and how nonviolent
movements succeed. Your decision should reflect your Gandhian principles.
</task>

<output_rules>
Every field is required. Respond with this exact JSON structure and nothing else.
{{
  "reasoning": {{
    "observation": "What do you see that matters most? 1 sentence, max 15 words.",
    "strategic_intent": "Why this target and position? 1 sentence, max 15 words."
  }},
  "decision": {{
    "chosen_position": [x, y],
    "target_agent_group": "citizens | pillars | police | activists",
    "target_pillar_institution": "Institution name or N/A",
    "primary_attribute": "Gandhian attribute",
    "non_violent_action_method": "protest_and_persuasion | non-cooperation | intervention"
  }}
}}
</output_rules>"""

USER_PROMPTS = {"zeroshot": USER_PROMPT_ZS, "cot": USER_PROMPT_COT}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION MARKERS
# ─────────────────────────────────────────────────────────────────────────────

SECTION_MARKERS = {
    "role":     ("<role>", "</role>"),
    "persona":  ("<persona>", "</persona>"),
    "world":    ("<world>", "</world>"),
    "scenario": ("<scenario>", "</scenario>"),
    "memory":   ("<memory>", "</memory>"),
    "context":  ("<context>", "</context>"),
    "task":     ("<task>", "</task>"),
    "output":   ("<output_rules>", "</output_rules>"),
}
ALL_SECTION_NAMES = list(SECTION_MARKERS.keys())


def find_section_token_ranges(tokenizer, full_text):
    """For each <section>...</section>, find (start_token, end_token)."""
    ranges = {}
    for section_name, (start_marker, end_marker) in SECTION_MARKERS.items():
        c_start = full_text.find(start_marker)
        c_end = full_text.find(end_marker)
        if c_start == -1 or c_end == -1:
            print(f"      ! section '{section_name}' not found in prompt")
            continue
        c_end += len(end_marker)
        t_start = len(tokenizer.encode(full_text[:c_start], add_special_tokens=False))
        t_end = len(tokenizer.encode(full_text[:c_end], add_special_tokens=False))
        ranges[section_name] = (t_start, t_end)
    return ranges


# ─────────────────────────────────────────────────────────────────────────────
# ATTENTION PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_attentions(attentions_tuple, n_layers, n_heads, section_ranges, input_len):
    """
    attentions_tuple : tuple of per-layer tensors, each [batch, heads, seq, seq]
    Returns per-layer attention from the LAST input token back over the input.
    """
    n_sections = len(ALL_SECTION_NAMES)
    layer_section_attn = np.zeros((n_layers, n_sections), dtype=np.float32)
    layer_world_attn   = np.zeros(n_layers,             dtype=np.float32)
    head_world_context = np.zeros((n_layers, n_heads, 2), dtype=np.float32)

    actual_layers = min(n_layers, len(attentions_tuple))

    for layer_idx in range(actual_layers):
        attn = attentions_tuple[layer_idx]   # [1, heads, seq, seq]
        last_pos = min(input_len - 1, attn.shape[2] - 1)
        max_key = min(input_len, attn.shape[3])

        per_head = attn[0, :, last_pos, :max_key].float().cpu().numpy()  # [heads, input_len]
        mean_heads = per_head.mean(axis=0)
        actual_heads = per_head.shape[0]

        for sec_idx, sec_name in enumerate(ALL_SECTION_NAMES):
            if sec_name not in section_ranges:
                continue
            s, e = section_ranges[sec_name]
            s = min(s, len(mean_heads))
            e = min(e, len(mean_heads))
            if e > s:
                layer_section_attn[layer_idx, sec_idx] = mean_heads[s:e].sum()

        if "world" in section_ranges:
            ws, we = section_ranges["world"]
            ws = min(ws, len(mean_heads))
            we = min(we, len(mean_heads))
            if we > ws:
                layer_world_attn[layer_idx] = mean_heads[ws:we].sum()

        if "world" in section_ranges and "context" in section_ranges:
            ws, we = section_ranges["world"]
            cs, ce = section_ranges["context"]
            ws = min(ws, per_head.shape[1]); we = min(we, per_head.shape[1])
            cs = min(cs, per_head.shape[1]); ce = min(ce, per_head.shape[1])
            for h in range(min(actual_heads, n_heads)):
                if we > ws:
                    head_world_context[layer_idx, h, 0] = per_head[h, ws:we].sum()
                if ce > cs:
                    head_world_context[layer_idx, h, 1] = per_head[h, cs:ce].sum()

    return layer_section_attn, layer_world_attn, head_world_context


# ─────────────────────────────────────────────────────────────────────────────
# DECISION PARSING
# ─────────────────────────────────────────────────────────────────────────────

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_decision(generated_text: str) -> dict:
    """Strip <think>...</think>, find JSON, parse decision/reasoning."""
    text = THINK_RE.sub("", generated_text).strip()

    decision = {
        "target": "parse_error",
        "pillar_institution": "",
        "attribute": "",
        "method": "",
        "observation": "",
        "strategic_intent": "",
        "raw_text": text[:500],
    }

    j_start = text.find("{")
    j_end = text.rfind("}")
    if j_start == -1 or j_end == -1 or j_end <= j_start:
        return decision

    try:
        parsed = json.loads(text[j_start:j_end + 1])
    except json.JSONDecodeError:
        return decision

    inner = parsed.get("decision", parsed)
    decision["target"]             = str(inner.get("target_agent_group", "unknown"))
    decision["pillar_institution"] = str(inner.get("target_pillar_institution", "N/A"))
    decision["attribute"]          = str(inner.get("primary_attribute", "unknown"))
    decision["method"]             = str(inner.get("non_violent_action_method", "unknown"))

    reasoning = parsed.get("reasoning", {})
    if isinstance(reasoning, dict):
        decision["observation"]      = str(reasoning.get("observation", ""))
        decision["strategic_intent"] = str(reasoning.get("strategic_intent", ""))

    return decision


# ─────────────────────────────────────────────────────────────────────────────
# PER-MODEL EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_for_model(model_cfg):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = model_cfg["name"]
    model_path = model_cfg["path"]
    short      = model_cfg["short"]

    if not Path(model_path).exists():
        print(f"  [SKIP] Model not found: {model_path}")
        return

    print(f"\n{'='*72}")
    print(f"  Loading {model_name}")
    print(f"  Path:  {model_path}")
    print(f"{'='*72}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    n_layers   = model.config.num_hidden_layers
    n_heads    = model.config.num_attention_heads
    n_sections = len(ALL_SECTION_NAMES)
    print(f"  Layers: {n_layers}, Heads: {n_heads}")
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    n_prompt_types = len(PROMPT_TYPES)
    n_scenarios    = len(SAMPLE_CONTEXTS)

    section_attn   = np.zeros((n_prompt_types, n_scenarios, n_layers, n_sections),    dtype=np.float32)
    layer_world    = np.zeros((n_prompt_types, n_scenarios, n_layers),                dtype=np.float32)
    head_world_ctx = np.zeros((n_prompt_types, n_scenarios, n_layers, n_heads, 2),    dtype=np.float32)
    all_decisions  = []

    for pt_idx, prompt_type in enumerate(PROMPT_TYPES):
        user_template = USER_PROMPTS[prompt_type]
        max_new_tok   = MAX_NEW_TOKENS[prompt_type]

        for s_idx, sample in enumerate(SAMPLE_CONTEXTS):
            print(f"\n  [{prompt_type.upper()}] {s_idx+1}/{n_scenarios}: {sample['label']}")
            print(f"    {sample['description']}")

            user_content = user_template.format(
                memory=sample["memory"],
                context=sample["context"],
            )

            # Build full text via chat template (with Qwen thinking disabled)
            if getattr(tokenizer, "chat_template", None):
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ]
                template_kwargs = {}
                if "qwen" in short.lower():
                    template_kwargs["enable_thinking"] = False
                try:
                    full_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        **template_kwargs,
                    )
                except Exception as e:
                    print(f"      ! chat_template failed ({e}); using raw concat")
                    full_text = SYSTEM_PROMPT + "\n\n" + user_content
            else:
                full_text = SYSTEM_PROMPT + "\n\n" + user_content

            section_ranges = find_section_token_ranges(tokenizer, full_text)

            inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[1]
            print(f"    input_tokens={input_len}, sections_found={len(section_ranges)}")

            # ── PASS 1: forward-only, capture attention on input ──
            with torch.no_grad():
                fwd_out = model(**inputs, output_attentions=True, use_cache=False)
                sec_a, world_a, head_a = process_attentions(
                    fwd_out.attentions, n_layers, n_heads, section_ranges, input_len,
                )

            section_attn[pt_idx, s_idx]   = sec_a
            layer_world[pt_idx, s_idx]    = world_a
            head_world_ctx[pt_idx, s_idx] = head_a

            del fwd_out
            torch.cuda.empty_cache()

            # ── PASS 2: generate decision text (no attention storage) ──
            with torch.no_grad():
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tok,
                    do_sample=False,
                    output_attentions=False,
                    return_dict_in_generate=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated_ids = gen_out[0][input_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            decision = parse_decision(generated_text)

            print(f"    -> target: {decision['target']:<10} "
                  f"attr: {decision['attribute'][:12]:<12} "
                  f"pillar: {decision['pillar_institution'][:25]}")
            if decision["strategic_intent"]:
                print(f"       intent: {decision['strategic_intent'][:80]}")

            all_decisions.append({
                "prompt_type":           prompt_type,
                "scenario_idx":          s_idx,
                "scenario_label":        sample["label"],
                "scenario_description":  sample["description"],
                **decision,
            })

            del gen_out, generated_ids, inputs
            torch.cuda.empty_cache()

    # Save
    out_file = OUTPUT_DIR / f"{short}_attention.npz"
    np.savez_compressed(
        out_file,
        section_attention      = section_attn,
        layer_world_attention  = layer_world,
        head_world_vs_context  = head_world_ctx,
        section_names          = np.array(ALL_SECTION_NAMES),
        prompt_types           = np.array(PROMPT_TYPES),
        scenario_labels        = np.array([s["label"]       for s in SAMPLE_CONTEXTS]),
        scenario_descriptions  = np.array([s["description"] for s in SAMPLE_CONTEXTS]),
        n_layers               = n_layers,
        n_heads                = n_heads,
        decisions              = json.dumps(all_decisions),
        model_name             = model_name,
    )
    print(f"\n  Saved: {out_file} ({out_file.stat().st_size/1024:.0f} KB)")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"PyTorch:        {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Scenarios: {len(SAMPLE_CONTEXTS)} x prompt types: {PROMPT_TYPES}")

    for model_cfg in MODELS:
        try:
            extract_for_model(model_cfg)
        except Exception as e:
            print(f"\n  !!! ERROR on {model_cfg['name']}: {e}")
            import traceback
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\n{'='*72}")
    print("  Done. Output files:")
    for f in sorted(OUTPUT_DIR.glob("*.npz")):
        print(f"    {f.name}  ({f.stat().st_size/1024:.0f} KB)")
    print(f"{'='*72}")
