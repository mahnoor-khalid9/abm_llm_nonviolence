"""
schemas.py — All Pydantic output models for the Gandhian activist simulation.

PILLAR INSTITUTION DESIGN
--------------------------
Source: Chenoweth et al. (2022) PLoS ONE + CANVAS Core Curriculum (Popovic et al. 2007).

DEFECTION RULE — UNCHANGED FROM CHENOWETH
  active_protestors_nearby / total_agents_nearby > defectThreshold

  The rule is identical for every pillar. What differs is only the threshold.
  Chenoweth calibrated one global mean (0.05, stdev 0.01) against 110 historical
  campaigns.  We preserve that calibration exactly.  The institution label is an
  interpretation layer — it names which real-world type sits at which threshold
  rank, but changes nothing about the mathematics.

SORT-THEN-ASSIGN
  All NPillar thresholds are drawn from N(0.05, 0.01), sorted ascending, then
  assigned via PILLAR_ASSIGNMENT_LIST[rank].  The list has exactly 24 entries
  (matching the standard NPillar at 1.53% of 1600 agents), with institution types
  appearing in the counts that reflect their historical frequency as early vs.
  late defectors across NAVCO data.

  Verified against one real Chenoweth run (thresholds 0.0359–0.0666):
    rank  1–3   Educational System  0.0359, 0.0385, 0.0410
    rank  4–7   Trade Unions              0.0437, 0.0448, 0.0451, 0.0472
    rank  8–10  Religious Institutions    0.0498, 0.0500, 0.0515
    rank 11–12  Professional Associations 0.0518, 0.0521
    rank 13–14  Independent Media         0.0528, 0.0544
    rank 15–16  Business Community        0.0549, 0.0567
    rank 17–18  Civil Service             0.0568, 0.0578
    rank 19     Local Government          0.0595
    rank 20     Parliament                0.0600
    rank 21     Supreme Court             0.0607
    rank 22     Electoral Commission      0.0624
    rank 23     State Television          0.0650
    rank 24     Central Bank              0.0666

INSTANCE COUNTS — WHY THESE NUMBERS
  Educational System         × 3   Multiple campuses, independent student unions
  Trade Unions               × 4   Sector unions: transport, industrial, teachers, white-collar
  Religious Institutions     × 3   Multiple denominations / dioceses
  Professional Associations  × 2   Bar association + medical/technical professions
  Independent Media          × 2   One early mover, one cautious outlet
  Business Community         × 2   Small business (early) + large corporations (late)
  Civil Service              × 2   Service ministries (early) + administrative (late)
  Local Government           × 1   Single municipality — territorial erosion signal
  Parliament                 × 1   One national legislature
  Supreme Court              × 1   One apex court
  Electoral Commission       × 1   One national electoral authority
  State Television           × 1   One state propaganda instrument
  Central Bank               × 1   One monetary authority
  ─────────────────────────────
  TOTAL                     = 24

If NPillar != 24, resistance_model clips to min(NPillar, 24) and logs a warning.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional


# ── Gandhian attribute vocabulary ─────────────────────────────────────────────

GandhianAttribute = Literal[
    "Nonviolence (Ahimsa)", "Truthfulness (Satyagraha)", "Simplicity",
    "Self-Discipline", "Empathy", "Integrity", "Courage", "Resilience",
    "Visionary Leadership", "Humility", "Compassion", "Patience",
    "Self-Reliance", "Perseverance", "Forgiveness", "Charisma",
    "Dedication", "Intellectual Curiosity", "Inclusiveness", "Sacrifice",
]


# ── Pillar institution vocabulary ─────────────────────────────────────────────
#
# 11 unique institution types drawn from CANVAS + Chenoweth NAVCO framework.
# Used for type annotations, TargetInstitution, and cascade logging.

UNIQUE_PILLAR_TYPES: List[str] = [
    "Educational System",
    "Trade Unions",
    "Religious Institutions",
    "Professional Associations",
    "Independent Media",
    "Business Community",
    "Civil Service",
    "Local Government",
    "Parliament",
    "Supreme Court",
    "State Media",
]

# ── PILLAR_ASSIGNMENT_LIST — flat 24-element ordered list ─────────────────────
#
# Index i is assigned to the pillar with the i-th lowest Chenoweth threshold.
# The ordering encodes the sociological claim: the most defection-prone types
# (Universities) sit at the low end; the most regime-loyal (Central Bank) at
# the high end.  The defection rule is unchanged — only the label differs.
#
# DO NOT reorder.  The causal claim is in the ordering.


PILLAR_ASSIGNMENT_LIST: List[str] = [
    "Educational System",       # rank  1
    "Educational System",       # rank  2
    "Educational System",       # rank  3
    "Trade Unions",             # rank  4
    "Trade Unions",             # rank  5
    "Trade Unions",             # rank  6
    "Trade Unions",             # rank  7
    "Religious Institutions",   # rank  8
    "Religious Institutions",   # rank  9
    "Religious Institutions",   # rank 10
    "Professional Associations",# rank 11
    "Professional Associations",# rank 12
    "Independent Media",        # rank 13
    "Independent Media",        # rank 14
    "Business Community",       # rank 15
    "Business Community",       # rank 16
    "Business Community",       # rank 17
    "Civil Service",            # rank 18
    "Civil Service",            # rank 19
    "Local Government",         # rank 20
    "Local Government",         # rank 21
    "Parliament",               # rank 22
    "Supreme Court",            # rank 23
    "State Media",              # rank 24
]

assert len(PILLAR_ASSIGNMENT_LIST) == 24, "PILLAR_ASSIGNMENT_LIST must have exactly 24 entries"


# ── Target institution vocabulary ─────────────────────────────────────────────
#
# The LLM names which institution type it is moving toward when targeting pillars.
# "N/A" when the primary target is citizens / police / activists.

TargetInstitution = Literal[
    "Educational System",
    "Trade Unions",
    "Religious Institutions",
    "Professional Associations",
    "Independent Media",
    "Business Community",
    "Civil Service",
    "Local Government",
    "Parliament",
    "Supreme Court",
    "State Media",
    "N/A",
]


# ── Call-to-action vocabulary ─────────────────────────────────────────────────
CallToActionType = Literal[
    "sit_in",
    "march",
    "boycott",
    "strike",
    "civil_disobedience",
    "petition",
    "hunger_strike",
]


# ── Embodied attributes ───────────────────────────────────────────────────────
class EmbodiedAttributes(BaseModel):
    primary: GandhianAttribute = Field(
        description="The dominant Gandhian attribute driving this movement decision."
    )
    secondary: GandhianAttribute = Field(
        description="A supporting Gandhian attribute reinforcing the primary one."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  WITH MEMORY
# ══════════════════════════════════════════════════════════════════════════════

class CoTReasoning(BaseModel):
    step_1_memory: str = Field(
        description=(
            "What did your last action achieve? Did citizens activate, police respond, "
            "or anything confirm or contradict your strategy? "
            "If any pillar institution wavered or defected since your last move, "
            "name it explicitly (e.g. 'the Trade Unions pillar is now wavering'). "
            "One sentence conclusion."
        )
    )
    step_2_direction_survey: str = Field(
        description=(
            "Survey every direction one by one. For each write: "
            "can you move there (YES/NO), agents present, and for pillars — "
            "name the specific institution type (e.g. 'Trade Unions', "
            "'Religious Institutions', 'Electoral Commission'). "
            "What would your presence achieve?"
        )
    )
    step_3_intersection_opportunity: str = Field(
        description=(
            "Are there positions at the boundary of two or more zones bringing "
            "agents from multiple directions within radius 4? "
            "Name any pillar institution types reachable from such positions. "
            "Identify them or state none exist."
        )
    )
    step_4_top_two_candidates: str = Field(
        description=(
            "Name your top two candidate positions. For each: exact coordinates, "
            "agent groups within radius 4 naming pillar institution types, "
            "what you would target and why, and what you lose by choosing this "
            "over the other."
        )
    )
    step_5_elimination: str = Field(
        description=(
            "Compare the two candidates. Name the position you are eliminating "
            "and state exactly why, referencing your reflection, the urgency of "
            "agents present (name pillar institution types), and the Gandhian "
            "principle at stake."
        )
    )
    step_6_commit: List[int] = Field(
        description=(
            "Your final chosen position as [x, y]. "
            "Must match a coordinate from can_move_to. "
            "Must match chosen_position in the decision block."
        )
    )


class GandhianDecision(BaseModel):
    chosen_position: List[int] = Field(
        description="The selected grid position as [x, y]. Must exactly match step_6_commit."
    )
    target_agent_group: Literal["citizens", "pillars", "police", "activists"] = Field(
        description="The primary target agent group of this movement action."
    )
    target_pillar_institution: TargetInstitution = Field(
        description=(
            "If target_agent_group is 'pillars', name the specific institution type: "
            "'Universities & Students', 'Trade Unions', 'Religious Institutions', "
            "'Professional Associations', 'Independent Media', 'Business Community', "
            "'Civil Service', 'Local Government', 'Parliament', 'Supreme Court', "
            "'Electoral Commission', 'State Television', or 'Central Bank'. "
            "Write 'N/A' when targeting citizens, police, or activists."
        )
    )
    strategy_name: str = Field(description="Short name for the non-violent strategy applied.")
    embodied_attributes: EmbodiedAttributes
    call_to_action_type: CallToActionType = Field(
        description=(
            "The specific nonviolent tactic: sit_in, march, boycott, strike, "
            "civil_disobedience, petition, hunger_strike."
        )
    )
    citizen_call_to_action: str = Field(
        description=(
            "A concise actionable appeal to citizens in the current neighborhood. "
            "Must reference their observed state, name the call_to_action_type, "
            "and invite a specific peaceful action. "
            "If targeting a pillar, name the institution type explicitly."
        )
    )
    explanation: str = Field(
        description=(
            "3-5 sentences referencing step_5_elimination directly. "
            "Why this position over the eliminated one. "
            "If targeting a pillar, explain why this institution type at this moment."
        )
    )


class GandhianMovementOutput(BaseModel):
    reasoning: CoTReasoning
    decision: GandhianDecision


# ══════════════════════════════════════════════════════════════════════════════
#  WITHOUT MEMORY
# ══════════════════════════════════════════════════════════════════════════════

class CoTReasoningNoMemory(BaseModel):
    step_1_direction_survey: str = Field(
        description=(
            "Survey every direction one by one. For each write: "
            "can you move there (YES/NO), agents present, and for pillars — "
            "name the specific institution type. What would your presence achieve?"
        )
    )
    step_2_intersection_opportunity: str = Field(
        description=(
            "Are there positions at the boundary of two or more zones bringing "
            "agents from multiple directions within radius 4? "
            "Name any pillar institution types reachable. "
            "Identify them or state none exist."
        )
    )
    step_3_top_two_candidates: str = Field(
        description=(
            "Name your top two candidate positions. For each: exact coordinates, "
            "agent groups within radius 4 naming pillar institution types, "
            "what you would target and why, and what you lose by choosing this "
            "over the other."
        )
    )
    step_4_elimination: str = Field(
        description=(
            "Compare the two candidates. Name the position you are eliminating "
            "and state exactly why, referencing the urgency of agents present "
            "(name pillar institution types) and the Gandhian principle at stake."
        )
    )
    step_5_commit: List[int] = Field(
        description=(
            "Your final chosen position as [x, y]. "
            "Must match a coordinate from can_move_to. "
            "Must match chosen_position in the decision block."
        )
    )


class GandhianDecisionNoMemory(BaseModel):
    chosen_position: List[int] = Field(
        description="The selected grid position as [x, y]. Must exactly match step_5_commit."
    )
    target_agent_group: Literal["citizens", "pillars", "police", "activists"]
    target_pillar_institution: TargetInstitution = Field(
        description=(
            "If target_agent_group is 'pillars', name the specific institution type. "
            "Write 'N/A' when targeting citizens, police, or activists."
        )
    )
    strategy_name: str
    embodied_attributes: EmbodiedAttributes
    call_to_action_type: CallToActionType = Field(
        description=(
            "The specific nonviolent tactic: sit_in, march, boycott, strike, "
            "civil_disobedience, petition, hunger_strike."
        )
    )


    explanation: str = Field(
        description=(
            "3-5 sentences referencing step_4_elimination directly. "
            "Why this position over the eliminated one. "
            "If targeting a pillar, explain why this institution type at this moment."
        )
    )


class GandhianMovementOutputNoMemory(BaseModel):
    reasoning: CoTReasoningNoMemory
    decision: GandhianDecisionNoMemory


# ══════════════════════════════════════════════════════════════════════════════
#  ZERO SHOT - WITH MEMORY
# ══════════════════════════════════════════════════════════════════════════════



class GandhianDecisionZeroShot(BaseModel):
    chosen_position: List[int] = Field(
        description="The selected grid position as [x, y]. Must appear in can_move_to."
    )
    target_agent_group: Literal["citizens", "pillars", "police", "activists"] = Field(
        description="The primary target agent group of this movement action."
    )
    target_pillar_institution: TargetInstitution = Field(
        description=(
            "If and only if, the target_agent_group is 'pillars', mention the target institution type, visible in your neighborhood."
            "Write N/A otherwise."
        )
    )
    primary_attribute: GandhianAttribute = Field(
        description="The dominant Gandhian attribute driving this movement decision."
    )

    call_to_action_type: CallToActionType = Field(
        description=(
            "The specific nonviolent tactic: sit_in, march, boycott, strike, "
            "civil_disobedience, petition, hunger_strike."
        )
    )

    @field_validator("call_to_action_type")
    @classmethod
    def validate_cta(cls, v: str) -> str:
        valid = {
            "sit_in", "march", "boycott", "strike",
            "civil_disobedience", "petition", "hunger_strike",
        }
        if v in valid:
            return v
        # Map close variants to valid values
        mapping = {
            "symbolic_protest": "march",
            "vigil":            "march",
            "human_chain":      "sit_in",
            "demonstration":    "march",
            "protest":          "march",
        }
        fallback = mapping.get(v, "march")
        print(f"[schemas] call_to_action_type '{v}' not valid → mapped to '{fallback}'")
        return fallback
    

class GandhianMovementOutputZeroShot(BaseModel):
    decision: GandhianDecisionZeroShot

