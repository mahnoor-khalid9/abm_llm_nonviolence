"""
nlp_memory.py

Deterministic memory generation for Gandhian activist agents.
Compact narratives using Chenowett/Gandhian vocabulary as LLM priming.

generate_gandhian_memory       → {{SELF_REFLECTION}}
generate_neighborhood_memory   → {{RECENT_EVENTS}}

No em-dashes or colons in narrative output.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def _safe_ctx(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


# ══════════════════════════════════════════════════════════════════════════════
#  RECENT EVENTS  →  {{RECENT_EVENTS}}
# ══════════════════════════════════════════════════════════════════════════════

def generate_neighborhood_memory(agent_id: int, memory_data: Dict) -> str:
    if not memory_data:
        return "First step. No prior memory."

    obs = memory_data.get("neighborhood_observations", {})
    lines = []

    v = _narrate_police_violence(obs)
    if v: lines.append(v)

    d = _narrate_police_defection(obs)
    if d: lines.append(d)

    p = _narrate_pillar_events(obs)
    if p: lines.append(p)

    c = _narrate_citizen_changes(obs)
    if c: lines.append(c)

    a = _narrate_activist_movement(obs)
    if a: lines.append(a)

    if not lines:
        return "Neighbourhood quiet, it is the pause before courage gathers itself."

    return "\n".join(lines)


# ── Police violence ───────────────────────────────────────────────────────────

def _narrate_police_violence(obs: Dict) -> str:
    police_acts = obs.get("police_actions", [])
    kills   = [p for p in police_acts if p.get("event") == "kill"]
    arrests = [p for p in police_acts if p.get("event") == "jail"]

    if not kills and not arrests:
        return ""

    parts = []

    if kills:
        ctx = _safe_ctx(kills[0].get("context", {}))
        target_type = ctx.get("target_type", "agent")
        n = len(kills)

        if target_type == "ActivistAgent":
            parts.append(
                f"{n} comrade(s) killed. Moral shock. "
                f"Sacrifice strengthens resolve." if n == 1 else
                f"{n} comrades killed. Moral shock. Each death hardens our purpose."
            )
        elif target_type == "CitizenAgent":
            parts.append(
                f"{n} unarmed citizen(s) killed. Moral shock. "
                f"Every witness now carries moral obligation." if n == 1 else
                f"{n} citizens killed. Moral shock. "
                f"Backfire begins. Violence creates witnesses and reasons to resist."
            )
        else:
            parts.append(f"{n} killed nearby. Moral shock.")

    if arrests:
        n = len(arrests)
        if n <= 2:
            parts.append(f"{n} arrested nearby. Fear spreads, so does outrage.")
        else:
            parts.append(f"{n} arrested in a sweep. Mass arrests reveal desperation, not strength.")

    return " ".join(parts)


# ── Police defection ──────────────────────────────────────────────────────────

def _narrate_police_defection(obs: Dict) -> str:
    police_acts = obs.get("police_actions", [])
    defections = [p for p in police_acts if p.get("event") == "defect"]

    if not defections:
        return ""

    n = len(defections)
    if n == 1:
        return "A police officer defected. Regime grip weakens. Conscience over command."
    return f"{n} police defected. Regime grip weakens. Coercive capacity eroding."


# ── Pillar events ─────────────────────────────────────────────────────────────

_PILLAR_FRAMING = {
    "Religious Institution":    "Moral authority speaks. Conscience leaves the regime.",
    "Judiciary Institution":    "Law recoils from injustice. Legitimacy drains.",
    "Legislative Institution":  "Lawmakers reject their own creation. Self-delegitimisation.",
    "Business Community":       "Economic support withdrawn. The end accelerates.",
    "Civil Service":            "State machinery slows. Operators refuse the gears.",
    "Educational System":       "Teachers declared truth. Students watching. Narrative collapses.",
    "Local Government":         "Consent fracturing. Local power chooses community over command.",
    "State-Supported Media":    "Propaganda cracks. State media will not carry the lie.",
}


def _narrate_pillar_events(obs: Dict) -> str:
    pillar_changes = obs.get("pillar_changes", [])
    defections = [p for p in pillar_changes if p.get("event") == "defect"]

    if not defections:
        return ""

    if len(defections) == 1:
        d = defections[0]
        ctx = _safe_ctx(d.get("context", {}))
        inst = ctx.get("institution_type",
                       d.get("institution_type", "an institution"))
        framing = _PILLAR_FRAMING.get(inst, "Our methods are working.")
        return f"{inst} defected. Institutional support crumbling. {framing}"

    inst_names = []
    for d in defections:
        ctx = _safe_ctx(d.get("context", {}))
        inst_names.append(ctx.get("institution_type",
                                  d.get("institution_type", "institution")))

    return (
        f"{len(defections)} pillars defected ({', '.join(inst_names)}). "
        f"Cascade may be forming. Institutional support crumbling."
    )


# ── Citizen changes ───────────────────────────────────────────────────────────

def _narrate_citizen_changes(obs: Dict) -> str:
    citizen_acts = obs.get("citizen_activity", [])

    joined = [c for c in citizen_acts if c.get("event") in ("protest_start", "active_nv")]
    jailed = [c for c in citizen_acts if c.get("event") == "jail"]
    killed = [c for c in citizen_acts if c.get("event") == "kill"]

    if not joined and not jailed and not killed:
        return ""

    parts = []

    if joined:
        n = len(joined)
        if n == 1:
            parts.append("+1 joined the movement. Courage witnessed inspires courage.")
        elif n <= 3:
            parts.append(f"+{n} joined the movement. Conscience spreading.")
        else:
            parts.append(f"+{n} joined the movement. Critical mass forming.")

    if killed:
        n = len(killed)
        parts.append(f"{n} citizen(s) killed. Backfire. Violence begins movements, not ends them.")

    if jailed:
        n = len(jailed)
        parts.append(f"{n} citizen(s) arrested. Regime creates witnesses.")

    return " ".join(parts)


# ── Activist coordination ────────────────────────────────────────────────────

def _narrate_activist_movement(obs: Dict) -> str:
    others = obs.get("other_activists", [])
    if not others:
        return ""

    active = [a for a in others if a.get("event") in
              ("movement_decision", "protest_start", "active_nonviolent", "active_nv")]
    jailed = [a for a in others if a.get("event") == "jail"]
    killed = [a for a in others if a.get("event") == "kill"]

    if not active and not jailed and not killed:
        return ""

    parts = []

    if active:
        n = len(active)
        if n == 1:
            parts.append("A fellow activist nearby. Coordination possible.")
        else:
            parts.append(f"{n} activists are active nearby. We amplify each other.")

    if killed:
        n = len(killed)
        parts.append(f"{n} comrade(s) killed. We carry their choice forward.")

    if jailed:
        n = len(jailed)
        parts.append(f"{n} comrade(s) arrested. We regroup and persist.")

    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  SELF REFLECTION  →  {{SELF_REFLECTION}}
# ══════════════════════════════════════════════════════════════════════════════

_INTENT = {
    "citizens":  "to stir the conscience already alive within them",
    "pillars":   "to place before them the moral cost of complicity",
    "police":    "to remind them they too have a choice",
    "activists": "to strengthen our shared resolve",
}

_ATTRIBUTE = {
    "Leadership":  "bearing and presence",
    "Courage":     "visible fearlessness",
    "Empathy":     "compassionate witness",
    "Vision":      "articulating what could be",
    "Resilience":  "stubborn refusal to break",
    "Faith":       "unshaken conviction",
}

_METHOD_LABEL = {
    "protest_and_persuasion": "protest and persuasion",
    "non-cooperation":       "non-cooperation",
    "intervention":          "intervention",
}


def generate_gandhian_memory(agent_id: int, memory_data: Dict) -> str:
    actions = memory_data.get("own_actions", [])
    if not actions:
        return "I held my ground in quiet witness. Present, steady, rooted."

    is_killed = False
    is_jailed = False
    jail_term = 0
    is_active_nv = False
    movement_line = ""
    protest_declared = False
    unknown_events: List[str] = []

    for act in actions:
        event = act.get("event", "")
        ctx = _safe_ctx(act.get("context", {}))

        if event == "kill":
            is_killed = True
        elif event == "jail":
            is_jailed = True
            jail_term = int(ctx.get("jailTerm_left", 0))
        elif event == "active_nonviolent":
            is_active_nv = True
        elif event == "movement_decision":
            target = (act.get("target") or "community").lower()
            attribute = act.get("primary_attribute") or "Leadership"
            method = act.get("non_violent_action_method") or "nonviolent action"
            method = _METHOD_LABEL.get(method, method)

            intent = _INTENT.get(target, "to serve the cause of truth")
            attr_phrase = _ATTRIBUTE.get(attribute, attribute.lower())

            movement_line = (
                f"Guided by {attr_phrase}, I approached {target} "
                f"through {method}, {intent}."
            )
        elif event == "protest_start":
            protest_declared = True
        else:
            unknown_events.append(event)

    # DEATH
    if is_killed:
        return (
            "I was killed. The cause endures beyond me. "
            "Let my death be proof they have no answer except violence."
        )

    # IMPRISONMENT
    if is_jailed:
        if jail_term <= 2:
            return "I am imprisoned but release is near. I endure. The truth struck home."
        elif jail_term <= 5:
            return (
                f"I am imprisoned for {jail_term} steps. I endure. "
                f"Fellow activists, do not wait near me. Go where you are needed."
            )
        else:
            return (
                f"I am imprisoned for {jail_term} steps. I endure. "
                f"Fellow activists, do not wait near me. "
                f"No one is indispensable. My comrades know what to do."
            )

    parts = []

    if is_active_nv:
        parts.append("I stand firm in active nonviolent resistance.")

    if movement_line:
        parts.append(movement_line)

    if protest_declared:
        parts.append("I stepped into public protest. Dissent made visible.")

    if is_active_nv and not movement_line and not protest_declared:
        parts.append("I hold this ground. Present, visible, committed.")

    if unknown_events:
        parts.append(f"[Unrecognised. {', '.join(set(unknown_events))}]")

    return " ".join(parts) if parts else "I held my ground in silent witness."