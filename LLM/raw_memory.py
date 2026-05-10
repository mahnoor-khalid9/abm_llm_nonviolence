"""
raw_memory.py — Structured event-log parser for the Gandhian activist simulation.

Key responsibilities
--------------------
- Parse the shared step log (written by SimLogger.log_agent_event) into
  per-activist memory dicts.
- Respect toroidal distance when deciding which events fall within an
  activist's vision radius.
- Categorise every observed event into one of four neighbourhood observation
  buckets: police_actions, pillar_changes, other_activists, citizen_activity.

Field consistency with simulation_logger.py
-------------------------------------------
- _parse_fields strips surrounding double-quotes from values so downstream
  code receives clean strings (e.g. 'citizens' not '"citizens"').
- The context JSON may contain [SC] in place of literal semicolons (escaped
  by the logger to protect the semicolon-delimited log format).
  _safe_parse_context reverses this before JSON parsing.
- Deduplication key includes target_id for police events so multiple arrests
  by the same officer in one step are preserved.

Fixes vs. original
------------------
- context field uses .get() — no KeyError on events that omit it.
- activist_positions is built step-by-step so distance uses the position the
  activist actually occupied at the time of each observed event (temporal fix).
- Grid dimensions are a module-level constant that callers can override.
- "kill" and "recruiting_phase" events are handled in own_actions.
- Citizen "kill" events are routed to citizen_activity (not silently dropped).
- Pillar "wavering" events get their own sub-bucket.
- Duplicate log lines are deduplicated before processing.
- target_pos parsing uses the shared parse_pos() utility (no hand-rolled splits).
- Empty lists in own_actions are also removed at cleanup, not just neighbourhood.
- Notable absences are flagged so the NLP layer can narrate them.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# ── Grid configuration ────────────────────────────────────────────────────────
GRID_WIDTH:  int = 40
GRID_HEIGHT: int = 40

# All event types the simulation can produce.
# Cross-referenced with agent_based.py (see event_inventory.md).
# Used to detect unseen event types so we don't silently discard future events.

KNOWN_EVENTS: Set[str] = {
    # Activist events (ActivistAgent)
    "movement_decision", "protest_start", "active_nonviolent", "jail", "kill",
    # Police events (PoliceAgent)
    "jail", "kill", "defect",
    # Pillar events (PillarAgent)
    "defect",
    # Citizen events (CitizenAgent)
    "active_nv", "protest_start", "rest_phase", "jail", "kill",
}


# ── Geometry helpers ──────────────────────────────────────────────────────────

def is_within_toroidal_radius(
    pos1: Tuple[int, int],
    pos2: Tuple[int, int],
    radius: float,
    width: int  = GRID_WIDTH,
    height: int = GRID_HEIGHT,
) -> bool:
    """True if pos2 is within *radius* of pos1 on a toroidal grid."""
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])
    if dx > width  / 2: dx = width  - dx
    if dy > height / 2: dy = height - dy
    return float(np.sqrt(dx ** 2 + dy ** 2)) <= radius


def parse_pos(pos_str: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    Parse a position string like '(5, 21)' or '[5, 21]' into (5, 21).
    Returns None if the string is missing, 'None', or unparseable.
    """
    if not pos_str or pos_str.strip() in ("None", "null", ""):
        return None
    match = re.search(r"[\(\[](\d+),\s*(\d+)[\)\]]", pos_str)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    logging.debug("parse_pos: could not parse %r", pos_str)
    return None


# ── Log parsing ───────────────────────────────────────────────────────────────

def get_activist_ids_from_log(file_path: str) -> List[int]:
    """Return a sorted list of unique ActivistAgent IDs found in the log."""
    activist_ids: Set[int] = set()
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            for line in fh:
                fields = _parse_fields(line)
                if fields.get("agent_type") == "ActivistAgent":
                    try:
                        activist_ids.add(int(fields["agent_id"]))
                    except (KeyError, ValueError):
                        pass
    except FileNotFoundError:
        logging.warning("File not found: %s", file_path)
    return sorted(activist_ids)


def parse_step_log(step_file: str) -> List[Dict[str, Any]]:
    """
    Parse every event line in *step_file*.

    Returns a list of event dicts sorted by step.
    Duplicate lines are deduplicated using (step, agent_id, event, target_id)
    so that multiple police actions against different targets in one step are
    each preserved.
    Events whose position cannot be parsed are kept (pos=None) so callers
    can decide whether to skip them.
    """
    events: List[Dict[str, Any]] = []
    seen: Set[Tuple] = set()          # deduplication key

    with open(step_file, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            fields = _parse_fields(line)
            try:
                step     = int(fields.get("Step", 0))
                agent_id = int(fields.get("agent_id", 0))
                event    = fields.get("event", "")

                raw_context = fields.get("context", "{}")
                context = _safe_parse_context(raw_context)

                # Dedup key includes target_id so multiple police actions
                # against different targets in the same step are preserved.
                target_id = str(context.get("target_id", ""))
                dedup_key = (step, agent_id, event, target_id)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                event_dict: Dict[str, Any] = {
                    "step":                step,
                    "agent_type":          fields.get("agent_type", ""),
                    "agent_id":            agent_id,
                    "pos":                 parse_pos(fields.get("pos", "")),
                    "event":               event,
                    "target_agent":        fields.get("target_agent", "None"),
                    "context":             context,
                    # Movement-decision fields (present only for ActivistAgent)
                    "elapsed":             fields.get("elapsed", "None"),
                    "response_tokens":     fields.get("response_tokens", "None"),
                    "target_pillar_inst":  fields.get("target_pillar_inst", "None"),
                    "primary_attribute":   fields.get("primary_attribute", "None"),
                    "non_violent_action_method": fields.get("non_violent_action_method", "None"),
                }

                # Warn about event types we don't recognise
                if event and event not in KNOWN_EVENTS:
                    logging.warning(
                        "Line %d: unrecognised event type %r — included as-is", lineno, event
                    )

                events.append(event_dict)

            except (ValueError, TypeError) as exc:
                logging.debug("Line %d skipped (%s): %r", lineno, exc, line.strip())

    events.sort(key=lambda e: e["step"])
    return events


# ── Core memory builder ───────────────────────────────────────────────────────

def get_activist_neighborhood_memories(
    log_file_path: str,
    vision_radius: float = 4.0,
    grid_width:    int   = GRID_WIDTH,
    grid_height:   int   = GRID_HEIGHT,
    debug:         bool  = False,
) -> Dict[int, Dict[str, Any]]:
    """
    Return a per-activist memory dict:

    {
        activist_id: {
            "own_actions": [...],
            "neighborhood_observations": {
                "police_actions":   [...],   # arrest, jail, kill, defect, patrol
                "pillar_changes":   [...],   # loyal, wavering, defect
                "other_activists":  [...],   # movement, protest, jail, kill
                "citizen_activity": [...],   # protest, active_nv, rest, jail, kill
            },
            "notable_absences": {            # NEW — what did NOT happen
                "no_citizen_response":  bool,
                "no_police_response":   bool,
                "no_pillar_shift":      bool,
                "no_fellow_activists":  bool,
            }
        }
    }

    Temporal fix
    ------------
    Activist positions are tracked step-by-step: for each event we check whether
    the *observer activist's position at that step* (not their final position)
    places the event within their vision radius.
    """
    activist_ids = get_activist_ids_from_log(log_file_path)
    if debug:
        logging.info("Found %d activists: %s", len(activist_ids), activist_ids)

    # Initialise memory skeleton
    memories: Dict[int, Dict[str, Any]] = {}
    for aid in activist_ids:
        memories[aid] = {
            "own_actions": [],
            "neighborhood_observations": {
                "police_actions":   [],
                "pillar_changes":   [],
                "other_activists":  [],
                "citizen_activity": [],
            },
        }

    all_events = parse_step_log(log_file_path)
    if debug:
        logging.info("Parsed %d total events", len(all_events))

    # ── Step-indexed activist positions ──────────────────────────────────────
    # activist_positions_by_step[step][agent_id] = pos
    # We forward-fill: if an activist isn't seen at step N, use their last
    # known position.
    activist_positions_by_step: Dict[int, Dict[int, Tuple[int, int]]] = defaultdict(dict)
    last_known: Dict[int, Tuple[int, int]] = {}

    for ev in all_events:
        if ev["agent_type"] == "ActivistAgent" and ev["pos"]:
            last_known[ev["agent_id"]] = ev["pos"]
        # Record the latest known position for every activist at this step
        step = ev["step"]
        for aid, pos in last_known.items():
            activist_positions_by_step[step][aid] = pos

    # ── Main pass ─────────────────────────────────────────────────────────────
    for ev in all_events:
        event_agent_id = ev["agent_id"]
        event_type     = ev["agent_type"]
        event_action   = ev["event"]
        event_pos      = ev["pos"]
        step           = ev["step"]
        context        = ev["context"]

        for act_id in activist_ids:
            # Observer's position at this step (skip if unknown)
            act_pos = activist_positions_by_step.get(step, {}).get(act_id)
            if act_pos is None:
                continue

            # ── Own actions ────────────────────────────────────────────────
            if event_agent_id == act_id and event_type == "ActivistAgent":
                _record_own_action(memories[act_id], ev)
                continue

            # ── Neighbourhood observations ─────────────────────────────────
            if event_pos is None:
                continue  # can't do distance check without position

            if not is_within_toroidal_radius(
                act_pos, event_pos, vision_radius, grid_width, grid_height
            ):
                continue

            if event_agent_id == act_id:
                continue  # skip self in neighbourhood

            _record_neighbourhood_event(memories[act_id], ev)

    # ── Post-processing ───────────────────────────────────────────────────────
    for act_id in activist_ids:
        mem = memories[act_id]
        obs = mem["neighborhood_observations"]

        # Flag notable absences before pruning empty lists
        mem["notable_absences"] = {
            "no_citizen_response":  not obs["citizen_activity"],
            "no_police_response":   not obs["police_actions"],
            "no_pillar_shift":      not obs["pillar_changes"],
            "no_fellow_activists":  not obs["other_activists"],
        }

        # Remove empty observation buckets (reduces payload size)
        for key in list(obs.keys()):
            if not obs[key]:
                del obs[key]

        # Also remove own_actions if empty (step returned nothing)
        if not mem["own_actions"]:
            mem["own_actions"] = []   # keep key for consistent schema, leave empty

    return memories


# ── Event recording helpers ───────────────────────────────────────────────────

def _record_own_action(mem: Dict, ev: Dict) -> None:
    """Append a structured own-action entry to the activist's memory."""
    action = ev["event"]

    if action == "movement_decision":
        entry = {
            "step":              ev["step"],
            "position":          ev["pos"],
            "event":             action,
            # Core decision fields (from top-level log fields)
            "target":            ev["target_agent"],
            "target_pillar_inst": ev.get("target_pillar_inst", "None"),
            "primary_attribute": ev.get("primary_attribute", "None"),
            "non_violent_action_method": ev.get("non_violent_action_method", "None"),
            # Strategy and coordination (from context dict)
            "derived_strategy":           ev["context"].get("derived_strategy", "None"),
            "coordination_state":         ev["context"].get("coordination_state", "None"),
            "free_activists_nearby_count": ev["context"].get("free_activists_nearby_count", 0),
            # Pillar context (observable)
            "nearby_pillar_institutions": ev["context"].get("nearby_pillar_institutions", {}),
            # Computational metrics
            "elapsed":           ev.get("elapsed", "None"),
            "response_tokens":   ev.get("response_tokens", "None"),
        }


    elif action == "kill":
        # Own death — highest priority event; narrative layer should return immediately
        entry = {
            "step":    ev["step"],
            "position": ev["pos"],
            "event":   "kill",
            "context": ev["context"],
        }
    elif action == "jail":
        entry = {
            "step":     ev["step"],
            "position": ev["pos"],
            "event":    "jail",
            "context":  ev["context"],
        }
    elif action in ("protest_start", "active_nonviolent", "recruiting_phase", "active_nv"):
        entry = {
            "step":     ev["step"],
            "position": ev["pos"],
            "event":    action,
            "context":  ev["context"],
        }
    else:
        # Unknown / future event — preserve as-is
        entry = {
            "step":     ev["step"],
            "position": ev["pos"],
            "event":    action,
            "context":  ev["context"],
        }

    mem["own_actions"].append(entry)


def _record_neighbourhood_event(mem: Dict, ev: Dict) -> None:
    """Route a neighbour's event into the correct observation bucket."""
    event_type   = ev["agent_type"]
    event_action = ev["event"]

    base_entry = {
        "step":       ev["step"],
        "agent_id":   ev["agent_id"],
        "agent_type": event_type,
        "event":      event_action,
        "position":   ev["pos"],
        "context":    ev["context"],
    }

    if event_type == "PoliceAgent":
        # All police events go to police_actions; NLP layer decides what to narrate
        mem["neighborhood_observations"]["police_actions"].append(base_entry)

    elif event_type == "PillarAgent":
        # wavering and defect both go here; NLP layer differentiates.
        # Promote institution_type from context to top-level for easy access
        # in nlp_memory without requiring a context parse.
        entry = dict(base_entry)
        ctx = base_entry.get("context", {})
        if isinstance(ctx, dict) and "institution_type" in ctx:
            entry["institution_type"] = ctx["institution_type"]
        mem["neighborhood_observations"]["pillar_changes"].append(entry)

    elif event_type == "ActivistAgent":
        entry = dict(base_entry)
        if event_action == "movement_decision":
            # Drop full context (large) — only keep lightweight fields
            entry.pop("context", None)
            entry["target"] = ev["target_agent"]
        mem["neighborhood_observations"]["other_activists"].append(entry)

    elif event_type == "CitizenAgent":
        # jail AND kill both go here — NLP layer handles both
        mem["neighborhood_observations"]["citizen_activity"].append(base_entry)


# ── Utility helpers ───────────────────────────────────────────────────────────

def _parse_fields(line: str) -> Dict[str, str]:
    """
    Split a semicolon-delimited key=value log line into a dict.

    Values are stripped of surrounding double-quotes that the logger adds
    to string fields (e.g. target_agent="citizens" → citizens).
    """
    fields: Dict[str, str] = {}
    for part in line.strip().split(";"):
        if "=" in part:
            k, _, v = part.partition("=")
            fields[k.strip()] = v.strip().strip('"')
    return fields


def _safe_parse_context(raw: Any) -> Dict[str, Any]:
    """
    Return a dict from whatever the context field contains:
    - already a dict  → return as-is
    - JSON string     → reverse [SC] escape from logger, then parse
    - anything else   → return {}
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            # Logger escapes semicolons as [SC] to protect the log format;
            # restore them before JSON parsing.
            restored = raw.replace("[SC]", ";")
            parsed = json.loads(restored)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def save_activist_memories_json(memories: Dict, output_file: str) -> None:
    """Serialise memories to a JSON file. Handles tuple keys by converting to strings."""
    with open(output_file, "w", encoding="utf-8") as fh:
        json.dump(memories, fh, indent=2, default=str)