"""
nlp_memory.py — Memory generation for Gandhian activist agents.

Changes from previous version
------------------------------
generate_neighborhood_memory → RECENT_EVENTS structured format
  Previously produced a multi-paragraph prose narrative about the neighbourhood.
  Now produces a compact structured string with exactly four event fields:

    Police action  — kill or arrest of any agent nearby at T-1
    Pillar events  — any pillar defection (with institution type)
    Citizen changes — net new citizens who became nonviolently active
    Activist movement — other activists visible in the zone

  This maps to {{RECENT_EVENTS}} in the prompt template.
  Each field says "none" when nothing happened — consistent schema every step.

  Rationale: the LLM already receives the current neighborhood state in the
  context. What memory needs to convey is what *happened* at T-1 that caused
  the current state. Four event fields cover all causally relevant changes:
  - Police action: backfire mechanism, moral shock
  - Pillar events: cascade signal — which institution type just defected
  - Citizen changes: did the movement grow or shrink last step?
  - Activist movement: is coordination forming or dissolving?

generate_gandhian_memory — UNCHANGED
  Produces the activist's own-action self-reflection narrative.
  Maps to {{SELF_REFLECTION}} in the prompt template.
  Still needed: the activist must know its own state (jailed, active, etc.).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from LLM.raw_memory import parse_pos


# ── Context helper ────────────────────────────────────────────────────────────

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
#  RECENT EVENTS — structured T-1 event summary (maps to {{RECENT_EVENTS}})
# ══════════════════════════════════════════════════════════════════════════════

def generate_neighborhood_memory(agent_id: int, memory_data: Dict) -> str:
    """
    Produce a structured T-1 event summary for the activist's {{RECENT_EVENTS}}
    prompt variable.
    """
    if not memory_data:
        return "No memory available — first step or memory pipeline did not run."

    obs_data = memory_data.get("neighborhood_observations", {})
    agent_id = int(agent_id)

    police_action     = _extract_police_action(obs_data, agent_id)
    pillar_events     = _extract_pillar_events(obs_data)
    citizen_changes   = _extract_citizen_changes(obs_data)
    activist_movement = _extract_activist_movement(obs_data, agent_id)

    # If all four are empty — one consolidated message
    all_empty = all(
        v == "none"
        for v in [police_action, pillar_events, citizen_changes, activist_movement]
    )
    if all_empty:
        return "Last step: no change in state of any agent nearby — neighbourhood was quiet."

    # Otherwise include only the fields that have content
    lines = []
    if police_action != "none":
        lines.append(f"Police action: {police_action}")
    if pillar_events != "none":
        lines.append(f"Pillar events: {pillar_events}")
    if citizen_changes != "none":
        lines.append(f"Citizen changes: {citizen_changes}")
    if activist_movement != "none":
        lines.append(f"Activist movement: {activist_movement}")

    return "\n".join(lines)



def _extract_police_action(obs_data: Dict, agent_id: int) -> str:
    """
    Return the most significant police action at T-1.
    Priority: kill > arrest.  Returns "none" if no repression occurred.
    """
    police_acts = obs_data.get("police_actions", [])
    kills   = [p for p in police_acts if p.get("event") == "kill"]
    arrests = [p for p in police_acts if p.get("event") == "jail"]

    if kills:
        ctx = _safe_ctx(kills[0].get("context", {}))
        t_type = ctx.get("target_type", "agent")
        t_id   = ctx.get("target_id", "")
        t_pos  = ctx.get("target_pos", "")
        id_str = f" (id {t_id})" if t_id else ""
        pos_str = f" at {t_pos}" if t_pos else ""
        n_extra = len(kills) - 1
        extra   = f" (+{n_extra} more kill(s))" if n_extra > 0 else ""
        return f"{t_type}{id_str} killed{pos_str}{extra} — moral shock"

    if arrests:
        ctx = _safe_ctx(arrests[0].get("context", {}))
        t_type = ctx.get("target_type", "agent")
        t_id   = ctx.get("target_id", "")
        t_pos  = ctx.get("target_pos", "")
        term   = ctx.get("jail_term", "?")
        id_str = f" (id {t_id})" if t_id else ""
        pos_str = f" at {t_pos}" if t_pos else ""
        n_extra = len(arrests) - 1
        extra   = f" (+{n_extra} more arrest(s))" if n_extra > 0 else ""
        return f"{t_type}{id_str} arrested{pos_str}, jail_term={term}{extra}"

    return "none"


def _extract_pillar_events(obs_data: Dict) -> str:
    """
    Return defection events from nearby pillars at T-1.
    Includes institution type from the defection context so the activist
    knows specifically which CANVAS institution crossed the threshold.
    """
    pillar_changes = obs_data.get("pillar_changes", [])
    defections = [p for p in pillar_changes if p.get("event") == "defect"]

    if not defections:
        return "none"

    parts = []
    for d in defections:
        ctx  = _safe_ctx(d.get("context", {}))
        inst = ctx.get("institution_type", d.get("institution_type", "unknown institution"))
        pos  = d.get("position")
        pos_str = f" at {pos}" if pos else ""
        parts.append(f"{inst} defected{pos_str}")

    return "; ".join(parts)


def _extract_citizen_changes(obs_data: Dict) -> str:
    """
    Return the count of citizens who newly became active nonviolent at T-1.
    "Active" = protest_start or active_nv events.
    Returns "none" if no citizen activations occurred.
    """
    citizen_acts = obs_data.get("citizen_activity", [])

    new_active = [
        c for c in citizen_acts
        if c.get("event") in ("protest_start", "active_nv")
    ]
    jailed = [c for c in citizen_acts if c.get("event") == "jail"]
    killed = [c for c in citizen_acts if c.get("event") == "kill"]

    parts = []
    if new_active:
        n = len(new_active)
        parts.append(f"+{n} citizen(s) joined nonviolent action")
    if jailed:
        parts.append(f"{len(jailed)} citizen(s) arrested")
    if killed:
        parts.append(f"{len(killed)} citizen(s) killed")

    return "; ".join(parts) if parts else "none"


def _extract_activist_movement(obs_data: Dict, self_id: int) -> str:
    """
    Return a summary of other activists visible in the zone at T-1.
    Focuses on whether coordination partners are present.
    """
    other_activists = obs_data.get("other_activists", [])
    if not other_activists:
        return "none"

    active   = [a for a in other_activists if a.get("event") in ("movement_decision", "protest_start", "active_nonviolent", "active_nv")]
    jailed   = [a for a in other_activists if a.get("event") == "jail"]
    killed   = [a for a in other_activists if a.get("event") == "kill"]

    parts = []
    if active:
        n = len(active)
        parts.append(f"{n} activist(s) active nearby")
    if jailed:
        parts.append(f"{len(jailed)} activist(s) arrested")
    if killed:
        parts.append(f"{len(killed)} activist(s) killed")

    return "; ".join(parts) if parts else "none"


# ══════════════════════════════════════════════════════════════════════════════
#  SELF REFLECTION — own-action narrative (maps to {{SELF_REFLECTION}})
#  Unchanged from previous version.
# ══════════════════════════════════════════════════════════════════════════════

def generate_gandhian_memory(agent_id: int, memory_data: Dict) -> str:
    """
    Produce the activist's self narrative from their own_actions list.
    Maps to {{SELF_REFLECTION}} in the prompt template.

    Handled events
    --------------
    kill              — own death (returns immediately)
    jail              — imprisoned (returns immediately)
    active_nonviolent — currently in active NV phase
    movement_decision — which direction and why the activist moved
    protest_start     — began a public protest
    unknown           — preserved with a generic note (future-proof)
    """
    actions = memory_data.get("own_actions", [])
    if not actions:
        return "I remained in peaceful observation, staying loyal to the movement."

    is_killed        = False
    is_jailed        = False
    jail_duration    = ""
    is_activeNV      = False
    is_recruiting    = False
    recovery_duration = ""
    movement_intent  = ""
    protest_declared = False
    unknown_events:  List[str] = []

    STRATEGY_INTENT = {
        "citizens":  "awaken their conscience",
        "pillars":   "withdraw their support from the regime",
        "police":    "appeal to their humanity",
        "activists": "strengthen our unity",
    }

    def _to_duration(n: int) -> str:
        if n <= 3:  return "briefly"
        if n <= 7:  return "for an extended period"
        return "for a sustained period"

    for act in actions:
        event = act.get("event", "")
        ctx   = _safe_ctx(act.get("context", {}))

        if event == "kill":
            is_killed = True

        elif event == "jail":
            is_jailed     = True
            jail_duration = _to_duration(int(ctx.get("jailTerm_left", 0)))

        elif event == "active_nonviolent":
            is_activeNV       = True
            recovery_duration = _to_duration(int(ctx.get("StepsToEnd", 0)))

        elif event == "movement_decision":
            target = (act.get("target") or "community").lower()
            citizen_cta = act.get("citizen_cta")
            attribute  = act.get("primary_attribute") or "Leadership"
            intent = STRATEGY_INTENT.get(target, "deepen our commitment to truth")
            movement_intent = f"I moved toward {target} to {intent} via {attribute}"

        elif event == "protest_start":
            protest_declared = True

        else:
            unknown_events.append(event)

    narrative: List[str] = []

    if is_killed:
        return (
            "I was killed by the police. My time in this movement has ended, "
            "but the cause endures — let those who remain carry it forward."
        )

    if is_jailed:
        return (
            f"I am imprisoned {jail_duration} and cannot move, act, or influence anyone. "
            f"Any activist standing near me is wasting their position — "
            f"my presence offers nothing to the movement right now. "
            f"Fellow activists must move away and find where they are needed."
        )


    if is_activeNV:
        narrative.append("I am taking active part in this nonviolent movement.")

    if movement_intent:
        prefix = "Keeping this intent, " if (is_activeNV or is_recruiting) else ""
        narrative.append(f"{prefix}{movement_intent}.")

    if protest_declared:
        narrative.append("I stepped forward and began a public protest.")

    if is_activeNV and not movement_intent:
        narrative.append(
            "I am holding this space — "
            "present among those who suffer, visible and committed."
        )

    if unknown_events:
        narrative.append(
            f"[Note: unrecognised event(s) occurred this step: {', '.join(set(unknown_events))}]"
        )

    return " ".join(narrative) if narrative else "I held my ground in silent witness."
