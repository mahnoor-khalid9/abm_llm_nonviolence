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
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Literal, Optional, Any, Union, Dict
import json


import re
from dataclasses import dataclass

@dataclass
class LLMResponse:
    thinking: str | None    # content inside <think>...</think>
    content:  str     

# ── Gandhian attribute vocabulary ─────────────────────────────────────────────

GandhianAttribute = Literal[
    "Nonviolence (Ahimsa)", "Truthfulness (Satyagraha)", "Empathy", 
    "Courage", "Visionary Leadership", "Patience", "Perseverance", "Forgiveness", 
    "Inclusiveness", "Sacrifice",
]

# ── Pillar institution vocabulary ─────────────────────────────────────────────
#
# 11 unique institution types drawn from CANVAS + Chenoweth NAVCO framework.
# Used for type annotations, TargetInstitution, and cascade logging.

UNIQUE_PILLAR_TYPES: List[str] = [
    "Educational System",
    "Religious Institution",
    "Business Community",
    "Civil Service",
    "Local Government",
    "Judiciary Institution",             
    "Legislative Institution",          
    "State-Supported Media",   
]

# ── PILLAR_ASSIGNMENT_LIST — flat 24-element ordered list ─────────────────────
#
# Index i is assigned to the pillar with the i-th lowest Chenoweth threshold.
# The ordering encodes the sociological claim: the most defection-prone types
# (Universities) sit at the low end; the most regime-loyal (Central Bank) at
# the high end.  The defection rule is unchanged — only the label differs.
#
# DO NOT reorder.  The causal claim is in the ordering.

PILLAR_TYPE_WEIGHTS: Dict[str, float] = {  # Changed to float for flexibility
    "Educational System": 1.0,     
    "Religious Institution": 1.0, 
    "Business Community": 1.0,     
    "Civil Service": 1.0,          
    "Local Government": 1.0,       
    "Judiciary Institution": 1.0,             
    "Legislative Institution": 1.0,          
    "State-Supported Media": 1.0,            
}

assert len(PILLAR_TYPE_WEIGHTS) == 8, "PILLAR_TYPE_WEIGHTS must have exactly 8 pillar types"

# ── Target institution vocabulary ─────────────────────────────────────────────
#
# The LLM names which institution type it is moving toward when targeting pillars.
# "N/A" when the primary target is citizens / police / activists.

TargetInstitution = Literal[
    "Educational System",
    "Religious Institution",
    "Business Community",
    "Civil Service",
    "Local Government",
    "Judiciary Institution",             
    "Legislative Institution",          
    "State-Supported Media",  
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

NonViolentActionMethod = Literal[ 
    "protest_and_persuasion", 
    "non-cooperation", 
    "intervention" 
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



# ══════════════════════════════════════════════════════════════════════════════
#  WITH MEMORY
# ══════════════════════════════════════════════════════════════════════════════



# ── Step 1: Memory ────────────────────────────────────────────

LastInfluence = Literal[
    "none",
    "citizens_mobilized",
    "activists_clustered",
    "pillar_pressured",
    "police_triggered"
]

ChangeDetected = Literal[
    "none",
    "more_support",
    "more_risk",
    "mixed"
]

Coordination = Literal[
    "isolated",
    "supported"
]


class MemoryBlock(BaseModel):
    last_influence: Union[LastInfluence, str]
    change_detected: Union[ChangeDetected, str]
    coordination: Union[Coordination, str]


# ── Step 2: Survey ────────────────────────────────────────────
class DirectionSurvey(BaseModel):
    moves:     Optional[list[tuple[int, int]]] = None
    suffering: int = 0
    activists: int = 0
    # Changed from PillarInfo to a simple list of strings
    pillars:   Optional[list[str]] = None

    @field_validator("suffering", "activists", mode="before")
    @classmethod
    def coerce_int(cls, v: Any) -> int:
        if v is None or str(v).strip().lower() in ("", "null", "none"):
            return 0
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0

    @field_validator("moves", mode="before")
    @classmethod
    def coerce_moves(cls, v: Any) -> Optional[list[tuple[int, int]]]:
        if not v or (isinstance(v, list) and len(v) == 0):
            return None
        # Handle if LLM returns tuples or mixed formats
        result = []
        for item in v:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                result.append((int(item[0]), int(item[1])))
        return result if result else None


    @field_validator("pillars", mode="before")
    @classmethod
    def coerce_pillars(cls, v: Any) -> Optional[list[str]]:
        """
        Extracts just the list of institution names.
        Handles: 
        1. Simple list: ["Trade Unions"]
        2. String: '["Trade Unions"]' or "null"
        3. Nested Dict: {"stable": {"institutions": ["Trade Unions"]}}
        """
        if not v:
            return None

        # Handle string input (JSON or null-strings)
        if isinstance(v, str):
            v_clean = v.strip().lower()
            if v_clean in ("null", "none", ""):
                return None
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                return None

        # If it's the complex dictionary from your context, extract just the names
        if isinstance(v, dict):
            # Check for 'stable' key first
            stable_data = v.get("stable")
            if isinstance(stable_data, dict):
                return stable_data.get("institutions", [])
            # Fallback for other dict structures
            return v.get("institutions") if "institutions" in v else None

        # If it's already a list, return it
        if isinstance(v, list):
            return v if len(v) > 0 else None

        return None

class SurveyBlock(BaseModel):
    north:     Optional[DirectionSurvey] = None
    south:     Optional[DirectionSurvey] = None
    east:      Optional[DirectionSurvey] = None
    west:      Optional[DirectionSurvey] = None
    adjacent:  Optional[DirectionSurvey] = None
    northeast: Optional[DirectionSurvey] = None
    northwest: Optional[DirectionSurvey] = None
    southeast: Optional[DirectionSurvey] = None
    southwest: Optional[DirectionSurvey] = None


# ── Step 3: Position Resolution ───────────────────────────────

TargetGroup = Literal[
    "citizens",
    "activists",
    "pillars",
    "police"
]


class PositionBlock(BaseModel):
    best_move: tuple[int, int]
    target_group: Union[TargetGroup, str]
    pillar_required: bool

    @field_validator("pillar_required", mode="before")
    @classmethod
    def coerce_bool(cls, v: Any) -> bool:
        # "false" → False, "true" → True
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return bool(v)


# ── Full Reasoning Chain ──────────────────────────────────────

class CoTReasoning(BaseModel):
    step_1_memory:   MemoryBlock
    step_2_survey:   SurveyBlock
    step_3_position: PositionBlock


# ── Top-level Output ──────────────────────────────────────────

class GandhianMovementOutputCoT(BaseModel):
    reasoning: CoTReasoning
    decision:  GandhianDecisionZeroShot

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
    non_violent_action_method: NonViolentActionMethod = Field(
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
    target_pillar_institution: Union[TargetInstitution, str] = Field(
        description=(
            "If and only if, the target_agent_group is 'pillars', mention the target institution type, visible in your neighborhood."
            "Write 'N/A' otherwise."
        )
    )
    primary_attribute: Union[GandhianAttribute, str] = Field(
        description="The dominant Gandhian attribute driving this movement decision."
    )

    non_violent_action_method: Union[NonViolentActionMethod, str] = Field(
        description=(
            "The specific nonviolent method: protest_and_persuasion | non-cooperation | intervention."
        )
    )


    @field_validator("target_pillar_institution", mode="before")
    @classmethod
    def coerce_pillar_institution(cls, v: Any) -> str:
        if v is None or str(v).strip().lower() in ("null", "none", "n/a", ""):
            return "N/A"
        return v
    
    @field_validator("non_violent_action_method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        valid = {
            "protest_and_persuasion", 
            "non-cooperation", 
            "intervention" 
        }
        if v in valid:
            return v
        # Map close variants to valid values
        mapping = {
            "symbolic_protest": "protest_and_persuasion",
            "vigil":            "protest_and_persuasion",
            "human_chain":      "protest_and_persuasion",
            "demonstration":    "protest_and_persuasion",
            "protest":          "protest_and_persuasion",
        }
        fallback = mapping.get(v, "protest_and_persuasion")
        print(f"[schemas] non_violent_action_method '{v}' not valid → mapped to '{fallback}'")
        return fallback


    
class GandhianMovementOutputZeroShot(BaseModel):
    model_config = ConfigDict(extra='allow')
    # Modified for Gemini
    decision: GandhianDecisionGeminiZeroShot


class GandhianDecisionGeminiZeroShot(BaseModel):
    chosen_position: List[int] = Field(
        description="The selected grid position as [x, y]. Must appear in can_move_to."
    )
    target_agent_group: Literal["citizens", "pillars", "police", "activists"] = Field(
        description="The primary target agent group of this movement action."
    )
    target_pillar_institution: Union[TargetInstitution, str] = Field(
        description=(
            "If and only if, the target_agent_group is 'pillars', mention the target institution type, visible in your neighborhood."
            "Write 'N/A' otherwise."
        )
    )
    primary_attribute: Union[GandhianAttribute, str] = Field(
        description="The dominant Gandhian attribute driving this movement decision."
    )

    non_violent_action_method: Union[NonViolentActionMethod, str] = Field(
        description=(
            "The specific nonviolent method: protest_and_persuasion | non-cooperation | intervention."
        )
    )


    @field_validator("target_pillar_institution", mode="before")
    @classmethod
    def coerce_pillar_institution(cls, v: Any) -> str:
        if v is None or str(v).strip().lower() in ("null", "none", "n/a", ""):
            return "N/A"
        return v
    
    @field_validator("non_violent_action_method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        valid = {
            "protest_and_persuasion", 
            "non-cooperation", 
            "intervention" 
        }
        if v in valid:
            return v
        # Map close variants to valid values
        mapping = {
            "symbolic_protest": "protest_and_persuasion",
            "vigil":            "protest_and_persuasion",
            "human_chain":      "protest_and_persuasion",
            "demonstration":    "protest_and_persuasion",
            "protest":          "protest_and_persuasion",
        }
        fallback = mapping.get(v, "protest_and_persuasion")
        print(f"[schemas] non_violent_action_method '{v}' not valid → mapped to '{fallback}'")
        return fallback



    # situational_read: str = Field(
    #     description="A summary of the current situational context that influenced this decision."
    # )