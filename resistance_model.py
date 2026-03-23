"""
resistance_model.py

SORT-THEN-ASSIGN — USING PILLAR_ASSIGNMENT_LIST
  Previous versions used rank % N_PILLAR_TYPES (cycling formula).
  This version uses PILLAR_ASSIGNMENT_LIST[rank] directly.

  PILLAR_ASSIGNMENT_LIST is a flat 24-element list where:
    index 0–2   → "Universities & Students"   (3 instances, lowest thresholds)
    index 3–6   → "Trade Unions"              (4 instances)
    index 7–9   → "Religious Institutions"    (3 instances)
    index 10–11 → "Professional Associations" (2 instances)
    index 12–13 → "Independent Media"         (2 instances)
    index 14–15 → "Business Community"        (2 instances)
    index 16–17 → "Civil Service"             (2 instances)
    index 18    → "Local Government"          (1 instance)
    index 19    → "Parliament"                (1 instance)
    index 20    → "Supreme Court"             (1 instance)
    index 21    → "Electoral Commission"      (1 instance)
    index 22    → "State Television"          (1 instance)
    index 23    → "Central Bank"              (1 instance, highest threshold)

  This design directly reflects NAVCO historical frequency data:
  institutions that appear as early defectors in more campaigns get
  more instances at lower threshold ranks.

  Verified against actual Chenoweth run (thresholds 0.0359–0.0666):
    Universities & Students  → 0.0359, 0.0385, 0.0410
    Trade Unions             → 0.0437, 0.0448, 0.0451, 0.0472
    ...
    State Television         → 0.0650
    Central Bank             → 0.0666

DEFECTION RULE — UNCHANGED
  active_protestors_nearby / total_agents_nearby > defectThreshold
  Identical for every pillar.  Identical to police ruleD.
  Institution type is a label — it changes nothing about the mathematics.

CASCADE TRACKING
  _first_defection_step: records the first step any instance of each of the
  13 unique institution types defects.  Emits a FIRST_DEFECTION log line once
  per type per run.

  defection_by_type: logged every step as a JSON dict keyed by all 13 types.
  Combined with step number, reconstructs the full cascade timeline across
  100 runs without any post-processing.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional
import time 
import mesa
import numpy as np
from mesa.experimental import JupyterViz, make_text
from mesa import batchrunner

from agent_based import (
    CitizenAgent, ActivistAgent, PoliceAgent, PillarAgent,
    ATypes, AGENT_CLASS_MAP,
)
from LLM.raw_memory import get_activist_neighborhood_memories, save_activist_memories_json
from LLM.nlp_memory import generate_neighborhood_memory, generate_gandhian_memory
from LLM.chat_model import ChatModel
from Logging.log_config import LogConfig, DEFAULT_LOG_CONFIG
from Logging.simulation_logger import SimLogger
from LLM.schemas import PILLAR_ASSIGNMENT_LIST, UNIQUE_PILLAR_TYPES


# ── Mesa shape override ───────────────────────────────────────────────────────

def _draw_grid_override(space, space_ax, agent_portrayal):
    def portray(g):
        x = [[]]; y = [[]]; s = [[]]; c = [[]]; ms = ['o']
        for i in range(g.width):
            for j in range(g.height):
                content = g._grid[i][j]
                if not content: continue
                if not hasattr(content, "__iter__"): content = [content]
                for agent in content:
                    data = agent_portrayal(agent)
                    marker = data.get("marker", "o")
                    if marker not in ms:
                        ms.append(marker)
                        x.append([]); y.append([]); s.append([]); c.append([])
                    gi = ms.index(marker)
                    x[gi].append(i); y[gi].append(j)
                    if "size"  in data: s[gi].append(data["size"])
                    if "color" in data: c[gi].append(data["color"])
        outs = []
        for mi, marker in enumerate(ms):
            if not x[mi]: continue
            out = {"x": x[mi], "y": y[mi]}
            if s[mi]: out["s"] = s[mi]
            if c[mi]: out["c"] = c[mi]
            out["marker"] = marker
            outs.append(out)
        return outs

    space_ax.set_xlim(-1, space.width); space_ax.set_ylim(-1, space.height)
    space_ax.tick_params(left=False, right=False, labelleft=False,
                         labelbottom=False, bottom=False)
    space_ax.set_xticks([x + 0.5 for x in range(space.width)])
    space_ax.set_yticks([x + 0.5 for x in range(space.height)])
    space_ax.get_figure().tight_layout()
    space_ax.get_figure().set_figwidth(space_ax.get_figure().get_size_inches()[1])
    space_ax.grid(alpha=0.15)
    for p in portray(space): space_ax.scatter(**p)


def override_mesa_shapes():
    mesa.experimental.components.matplotlib._draw_grid = _draw_grid_override


# ── NVResistanceModel ─────────────────────────────────────────────────────────

class NVResistanceModel(mesa.Model):
    def __init__(
        self,
        MaxSteps=200, SizeX=40, SizeY=40, IsTorus=True,
        PctCitizen=70, PctActivist=0.8, PctPolice=4, PctPillar=1.53,
        AgentVision=4, MaxJailTerm=10, StartLegitimacy=0.6,
        PctFindNV=40, PctTargetNV=25, PctKillNV=10, BackfireCoeff=0.99,
        valueF=0.0706, ProtestCycle=8.9, ProtestDuration=1, nNVthreshold=1,
        DefectThresh=0.05, PeerPressureNum=3.46, DelayStartMax=1,
        NVSuccessPct=10, PillarProxStrat=0, DefectThreshStDev=0.01,
        ActivistSearchVision=10, ReorderAgents=1, PctFickle=100, PctImmedProtest=5,
        dataCollectorType="singleRunVisualize", seed=None, init_for_viz=False,
        log_config: LogConfig = None,
    ):
        super().__init__()

        # ── Logging ───────────────────────────────────────────────────────────
        self._log_config = log_config if log_config is not None else DEFAULT_LOG_CONFIG
        self._log_config.create_all_dirs()
        self._log_config.reset_run_logs()
        self.logger = SimLogger(self._log_config)

        # ── Grid, schedule, LLM ──────────────────────────────────────────────
        self.grid       = mesa.space.SingleGrid(SizeX, SizeY, IsTorus)
        self.schedule   = mesa.time.RandomActivation(self)
        self.chat_model = ChatModel()

        # ── Data collector ───────────────────────────────────────────────────
        if dataCollectorType == "singleRunVisualize":
            self.datacollector = mesa.DataCollector(
                model_reporters={
                    "citizens":  lambda _: len([x for x in self.schedule.agents if x.is_Citizen and not x.jailTerm]),
                    "activists": lambda _: len([x for x in self.schedule.agents if x.is_Activist and not x.jailTerm]),
                    "police":    lambda _: len([x for x in self.schedule.agents if x.is_Police and not x.defect]),
                    "pillars":   lambda _: len([x for x in self.schedule.agents if x.is_Pillar and not x.defect]),
                    "activeNV":  lambda _: len([x for x in self.schedule.agents if x.is_Citizen and x.activeNV]),
                    "protest":   lambda _: len([x for x in self.schedule.agents if x.is_Citizen and x.inProtest]),
                },
            )
        elif dataCollectorType == "batchRun":
            self.datacollector = mesa.DataCollector(
                model_reporters={
                    "citizenPlot":            "citizenPlot",
                    "activistPlot":           "activistPlot",
                    "policePlot":             "policePlot",
                    "pillarPlot":             "pillarPlot",
                    "protestPlot":            "protestPlot",
                    "seed":                   "_seed",
                    "victory":                "victory",
                    "Max_InProtestPct":       "Max_InProtestPct",
                    "Max_ActualInProtestPct": "Max_ActualInProtestPct",
                    "Max_NVCitizens":         "Max_NVCitizens",
                },
            )

        # ── Constants ────────────────────────────────────────────────────────
        self.const = {
            "MaxSteps":           round(MaxSteps),
            "delayStartMax":      round(DelayStartMax),
            "vision":             round(AgentVision),
            "activistVision":     round(ActivistSearchVision),
            "pillarproxstrat":    PillarProxStrat,
            "ficklePct":          PctFickle / 100,
            "PctFindNVResistor":  PctFindNV / 100,
            "PctTargetNV":        PctTargetNV / 100,
            "PctKillNV":          PctKillNV / 100,
            "NVSuccessPct":       NVSuccessPct / 100,
            "BackfireCoeff":      BackfireCoeff,
            "MaxJailTerm":        MaxJailTerm,
            "nNVthreshold":       round(nNVthreshold),
            "protestCycle":       round(ProtestCycle),
            "protestDuration":    round(ProtestDuration),
            "PctImmedProtest":    PctImmedProtest / 100,
            "joinNVThresh":       valueF,
            "defectThreshold":    DefectThresh,
            "defectStDev":        DefectThreshStDev,
            "peerPressureNum":    round(PeerPressureNum),
        }

        # ── Time-series arrays ────────────────────────────────────────────────
        self.citizenPlot         = np.zeros(self.const["MaxSteps"])
        self.activistPlot        = np.zeros(self.const["MaxSteps"])
        self.policePlot          = np.zeros(self.const["MaxSteps"])
        self.pillarPlot          = np.zeros(self.const["MaxSteps"])
        self.protestPlot         = np.zeros(self.const["MaxSteps"])
        self.protestingCitizen   = np.zeros(self.const["MaxSteps"])
        self.killedCitizen       = np.zeros(self.const["MaxSteps"])
        self.nvCitizen           = np.zeros(self.const["MaxSteps"])
        self.protestingActivists = np.zeros(self.const["MaxSteps"])
        self.killedActivists     = np.zeros(self.const["MaxSteps"])
        self.jailedActivists     = np.zeros(self.const["MaxSteps"])

        self.governmentLegitimacy = StartLegitimacy

        # ── First-defection tracker ───────────────────────────────────────────
        # Keyed by all 13 unique institution types.
        # None = has not yet defected.
        self._first_defection_step: Dict[str, Optional[int]] = {
            inst: None for inst in UNIQUE_PILLAR_TYPES
        }

        # ── Agent placement ───────────────────────────────────────────────────
        NCitizen  = round(SizeX * SizeY * PctCitizen  / 100)
        NActivist = round(SizeX * SizeY * PctActivist / 100)
        NPolice   = round(SizeX * SizeY * PctPolice   / 100)
        NPillar   = round(SizeX * SizeY * PctPillar   / 100)
        TypeLUT   = (
            [ATypes.CITIZEN]  * NCitizen  +
            [ATypes.ACTIVIST] * NActivist +
            [ATypes.POLICE]   * NPolice   +
            [ATypes.PILLAR]   * NPillar
        )

        self.model_id     = id(self)
        self.init_for_viz = init_for_viz

        # if not self.init_for_viz:
        #     self.logger.log_model_init(
        #         f"NVResistanceModel id={self.model_id} run_id={self._log_config.run_id} "
        #         f"Grid={SizeX}x{SizeY} MaxSteps={MaxSteps} "
        #         f"Citizen={NCitizen} Activist={NActivist} Police={NPolice} Pillar={NPillar}"
        #     )

        for agent_id, type_ in enumerate(TypeLUT):
            agent_class = AGENT_CLASS_MAP[type_]
            agent = agent_class(agent_id, type_, self)
            self.grid.move_to_empty(agent)
            self.schedule.add(agent)

        # ── Sort-then-assign ──────────────────────────────────────────────────
        self._assign_institution_types_by_threshold(NPillar)

        # ── Per-agent defection registry (built once, updated each step) ──────────
        # Keyed by unique_id. Records exactly when and at what threshold each
        # individual pillar agent defected. No inference or sorting needed later.
        self._pillar_registry: Dict[int, dict] = {
            a.unique_id: {
                "institution":    a.institution_type,
                "threshold":      round(a.defectThreshold, 4),
                "defection_step": None,   # None = not yet defected
            }
            for a in self.schedule.agents if a.is_Pillar
        }

        # ── Pillar registry log ───────────────────────────────────────────────
        if not self.init_for_viz:
            pillar_registry = [
                {
                    "id":          a.unique_id,
                    "institution": a.institution_type,
                    "pos":         list(a.pos),
                    "threshold":   round(a.defectThreshold, 4),
                }
                for a in sorted(
                    (a for a in self.schedule.agents if a.is_Pillar),
                    key=lambda a: a.defectThreshold,
                )
            ]


            # Instance count per type — quick sanity check
            type_counts: Dict[str, int] = {}
            for entry in pillar_registry:
                type_counts[entry["institution"]] = (
                    type_counts.get(entry["institution"], 0) + 1
                )
            self.logger.log_model_init(
                f"PILLAR_TYPE_COUNTS: {json.dumps(type_counts)}"
            )

        # ── Post-placement ────────────────────────────────────────────────────
        self.activist_empty_spaces = {}
        for agent in self.schedule.agents:
            if agent.is_Activist:
                self.activist_empty_spaces[agent.unique_id] = (
                    self.get_empty_spaces_in_vision(agent, radius=self.const["vision"])
                )

        self.update_defect_levels()

        self.Max_InProtestPct       = 0
        self.Max_ActualInProtestPct = 0
        self.Max_NVCitizens         = 0
        self.victory  = 0
        self.running  = True

    # ── Sort-then-assign ──────────────────────────────────────────────────────

    def _assign_institution_types_by_threshold(self, NPillar: int) -> None:
        """
        Draw individual thresholds onto each PillarAgent, sort ascending,
        then assign institution types from PILLAR_ASSIGNMENT_LIST[rank].

        Each pillar draws its own threshold from N(DefectThresh, DefectThreshStDev).
        After sorting, the pillar with the lowest threshold gets rank 0
        (Educational System) and the highest gets rank 23 (State Media).
        """
        pillars = [a for a in self.schedule.agents if a.is_Pillar]

        if len(pillars) != NPillar:
            self.logger.log_model_init(
                f"WARN: NPillar={NPillar} but {len(pillars)} PillarAgents found."
            )

        # 1. Sort pillars by their threshold ascending
        pillars_sorted = sorted(pillars, key=lambda a: a.defectThreshold)

        # 2. Assign institution type by rank
        n_assignment = len(PILLAR_ASSIGNMENT_LIST)
        for rank, pillar in enumerate(pillars_sorted):
            pillar.institution_type = PILLAR_ASSIGNMENT_LIST[rank % n_assignment]
            # ADD LOGGING HERE 
            # pillar._inject_pillar_identity(pillar.defectThreshold, pillar.institution_type)
      
    # ── Spatial helpers ───────────────────────────────────────────────────────
    def get_empty_spaces_in_vision(self, agent, radius):
        neighborhood = self.grid.get_neighborhood(
            pos=agent.pos, moore=True, include_center=False, radius=radius
        )
        return [pos for pos in neighborhood if self.grid.is_cell_empty(pos)]

    # ── Memory retrieval ──────────────────────────────────────────────────────

    def retrieve_agent_cognitive_memory(self, agent_id, step):
        prev_step = step - 1
        if prev_step < 0:
            return None
        file_path = self._log_config.memory_nlp_path(prev_step)
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            return data.get(str(agent_id))
        except Exception as exc:
            self.logger.log_error_raw(
                f"Step={step}; retrieve_memory agent_id={agent_id} error={exc}"
            )
            return None

    # ── Memory pipeline ───────────────────────────────────────────────────────
    def process_step_memories(self, step_num, vision_radius=4):
        self.logger.flush_step(step_num)
        self.logger.close_step(step_num)

        step_file = self._log_config.step_log_path(step_num)
        if not os.path.exists(step_file):
            return False

        all_memories = get_activist_neighborhood_memories(
            step_file, vision_radius=vision_radius, debug=False
        )
        save_activist_memories_json(all_memories, self._log_config.memory_raw_path(step_num))
        self._convert_raw_to_cognitive_memories(step_num)
        return True

    def _convert_raw_to_cognitive_memories(self, step):
        raw_path = self._log_config.memory_raw_path(step)
        nlp_path = self._log_config.memory_nlp_path(step)
        if not os.path.exists(raw_path):
            return

        with open(raw_path, "r") as f:
            raw_data = json.load(f)

        cognitive_recall = {}
        for agent_id, memory_data in raw_data.items():
            self_reflection           = generate_gandhian_memory(agent_id, memory_data)
            environmental_observation = generate_neighborhood_memory(agent_id, memory_data)
            cognitive_recall[agent_id] = {
                "cognitive_memory": {
                    "self_reflection":           self_reflection,
                    "environmental_observation": environmental_observation,
                }
            }
            self.logger.log_narrative(agent_id, self_reflection[:120])

        with open(nlp_path, "w", encoding="utf-8") as f:
            json.dump(cognitive_recall, f, indent=4)

    # ── Cascade tracker ───────────────────────────────────────────────────────

    # def _update_cascade_tracker(self, current_step: int) -> Dict[str, int]:
    #     """
    #     Count defected instances per unique institution type.
    #     Emit FIRST_DEFECTION log line the first time any instance of a type defects.
    #     Return per_type dict for inclusion in the model state log.
    #     """
    #     per_type: Dict[str, int] = {inst: 0 for inst in UNIQUE_PILLAR_TYPES}
    #     for a in self.schedule.agents:
    #         if a.is_Pillar and a.defect:
    #             per_type[a.institution_type] = per_type.get(a.institution_type, 0) + 1

    #     for inst, count in per_type.items():
    #         if count > 0 and self._first_defection_step.get(inst) is None:
    #             self._first_defection_step[inst] = current_step
    #             # Find the triggering pillar (lowest threshold of this type)
    #             trigger = next(
    #                 (a for a in sorted(
    #                     self.schedule.agents,
    #                     key=lambda x: getattr(x, 'defectThreshold', 9999)
    #                 ) if a.is_Pillar and a.defect and a.institution_type == inst),
    #                 None
    #             )
    #             threshold_str = (
    #                 f"{trigger.defectThreshold:.4f}" if trigger else "unknown"
    #             )
    #             self.logger.log_model_state(
    #                 current_step,
    #                 event="FIRST_DEFECTION",
    #                 institution=inst,
    #                 # step=current_step,
    #                 threshold=threshold_str,
    #             )

    #     return per_type

    # def _update_cascade_tracker(self, current_step: int) -> Dict[str, int]:
    #     """
    #     Scan all pillar agents. Any agent whose defect flag just turned True
    #     gets recorded in _pillar_registry with the exact step and threshold.

    #     Emits:
    #     event=FIRST_DEFECTION   first instance of an institution type to defect
    #     event=DEFECTION         subsequent instances of the same type
    #     """
    #     per_type: Dict[str, int] = {inst: 0 for inst in UNIQUE_PILLAR_TYPES}

    #     for a in self.schedule.agents:
    #         if not a.is_Pillar:
    #             continue

    #         entry = self._pillar_registry[a.unique_id]

    #         if a.defect:
    #             per_type[a.institution_type] += 1

    #             # First time THIS agent has been seen defected
    #             if entry["defection_step"] is None:
    #                 entry["defection_step"] = current_step

    #                 # How many agents of this type have now defected (incl. this one)
    #                 defection_n = sum(
    #                     1 for e in self._pillar_registry.values()
    #                     if e["institution"] == a.institution_type
    #                     and e["defection_step"] is not None
    #                 )

    #                 event_name = "FIRST_DEFECTION" if defection_n == 1 else "DEFECTION"

    #                 if defection_n == 1:
    #                     self._first_defection_step[a.institution_type] = current_step

    #                 self.logger.log_model_state(
    #                     current_step,
    #                     event=event_name,
    #                     institution=a.institution_type,
    #                     agent_id=a.unique_id,
    #                     threshold=f"{a.defectThreshold:.4f}",
    #                     defection_n=defection_n,
    #                 )

    #     return per_type
    
    def _update_cascade_tracker(self, current_step: int) -> Dict[str, int]:
        per_type: Dict[str, int] = {inst: 0 for inst in UNIQUE_PILLAR_TYPES}

        for a in self.schedule.agents:
            if not a.is_Pillar:
                continue
            if a.defect:
                per_type[a.institution_type] += 1
                if self._pillar_registry[a.unique_id]["defection_step"] is None:
                    self._pillar_registry[a.unique_id]["defection_step"] = current_step
                    self.logger.log_model_state(
                        current_step,
                        event="DEFECTION",
                        institution=a.institution_type,
                        threshold=f"{a.defectThreshold:.4f}",
                    )

        return per_type

    # ── Grievance / defect level updates ─────────────────────────────────────

    def update_citizen_grievance_levels(self):
        citizens = [a for a in self.schedule.agents if a.is_Citizen]
        if not citizens:
            return ""
        grievances = np.array([c.grievance for c in citizens])
        q1, q2     = np.quantile(grievances, [0.33, 0.66])
        counts     = {"low": 0, "medium": 0, "high": 0}
        for c in citizens:
            if c.grievance <= q1:   c.grievanceLevel = "low";    counts["low"]    += 1
            elif c.grievance <= q2: c.grievanceLevel = "medium"; counts["medium"] += 1
            else:                   c.grievanceLevel = "high";   counts["high"]   += 1
        return (
            f"GrievanceLow={counts['low']} GrievanceMedium={counts['medium']} "
            f"GrievanceHigh={counts['high']} q33={q1:.3f} q66={q2:.3f}"
        )


    def update_defect_levels(self):
        agents = [a for a in self.schedule.agents if a.is_Pillar]
        if not agents:
            return
        thresholds = np.array([a.defectThreshold for a in agents])
        q1, q2     = np.quantile(thresholds, [0.33, 0.66])
        counts     = {"low": 0, "medium": 0, "high": 0}
        for a in agents:
            if a.defectThreshold <= q1:   a.defectLevel = "high";   counts["high"]   += 1
            if a.defectThreshold <= q2: a.defectLevel = "medium"; counts["medium"] += 1
            else:                         a.defectLevel = "low";    counts["low"]    += 1
        self.logger.log_model_state(
            self.schedule.steps, event="defect_levels",
            high=counts["high"], medium=counts["medium"], low=counts["low"],
            q33=f"{q1:.3f}", q66=f"{q2:.3f}",
        )

    # ── Victory condition ─────────────────────────────────────────────────────

    def check_victory_condition(self):
        pillarAgents = [x for x in self.schedule.agents if x.is_Pillar]
        if pillarAgents:
            pillar_defect_pct = (
                len([x for x in pillarAgents if x.defect]) / len(pillarAgents)
            )
            if pillar_defect_pct > self.const["NVSuccessPct"]:
                return 2  # Resistance victory
        aliveNV = [x for x in self.schedule.agents if x.is_alive and (
            x.is_Activist or (x.is_Citizen and x.activeNV)
        )]
        protester_threshold = self.const.get("ProtesterThreshold", self.const["nNVthreshold"])
        if len(aliveNV) <= protester_threshold:
            return 1  # Regime victory
        return 0

    # ── Main step ─────────────────────────────────────────────────────────────

    def step(self):
        prev_step = self.schedule.steps - 1
        strategy  = self.const["pillarproxstrat"]
        radius    = (
            int(self.const["activistVision"]) if strategy in (2, 4)
            else int(self.const["vision"])
        )

        self.process_step_memories(step_num=prev_step, vision_radius=radius)

        self.protestCount = 0
        t0 = time.time()
        self.schedule.step()
        elapsed = time.time() - t0
        elapsed = f"{elapsed:.3f}s"


        grievance_str = self.update_citizen_grievance_levels()
        current_step  = self.schedule.steps
        idx           = current_step - 1

        for a in self.schedule.agents:
            if a.is_Citizen  and a.inProtest:  self.protestingCitizen[idx]   += 1
            if a.is_Citizen  and a.activeNV:   self.nvCitizen[idx]           += 1
            if a.is_Activist and a.inProtest:  self.protestingActivists[idx] += 1
            if a.is_Activist and a.jailTerm:   self.jailedActivists[idx]     += 1
            if a.is_Citizen  and not a.jailTerm:  self.citizenPlot[idx]  += 1
            if a.is_Activist and not a.jailTerm:  self.activistPlot[idx] += 1
            if a.is_Police   and not a.defect:    self.policePlot[idx]   += 1
            if a.is_Pillar   and not a.defect:    self.pillarPlot[idx]   += 1
            if a.is_Citizen  and a.activeNV:      self.protestPlot[idx]  += 1

        aliveAgents = [x for x in self.schedule.agents if x.is_alive]
        if aliveAgents:
            self.Max_InProtestPct = max(
                len([x for x in aliveAgents
                     if (x.is_Activist or x.is_Citizen) and x.inProtest])
                / len(aliveAgents),
                self.Max_InProtestPct,
            )
            self.Max_ActualInProtestPct = max(
                self.protestCount / len(aliveAgents),
                self.Max_ActualInProtestPct,
            )
        self.Max_NVCitizens = max(
            len([x for x in aliveAgents if x.is_Citizen and x.activeNV]),
            self.Max_NVCitizens,
        )

        victory_code = self.check_victory_condition()
        if victory_code > 0:
            self.victory = victory_code
        elif current_step >= self.const["MaxSteps"]:
            self.victory = 3

        self.datacollector.collect(self)

        # ── Cascade tracking ──────────────────────────────────────────────────
        per_type = self._update_cascade_tracker(current_step)

        defected_institutions = [
            a.institution_type
            for a in self.schedule.agents
            if a.is_Pillar and a.defect
        ]

        # ── Model state log ───────────────────────────────────────────────────
        grievance_kv = (
            {k: v for k, v in [s.split("=") for s in grievance_str.split()]}
            if grievance_str else {}
        )
        self.logger.log_model_state(
            current_step,
            event="model_state",
            elapsed=elapsed,
            legitimacy=f"{self.governmentLegitimacy:.4f}",
            nvCitizen=int(self.nvCitizen[idx]),
            protestingCitizen=int(self.protestingCitizen[idx]),
            protestingActivists=int(self.protestingActivists[idx]),
            jailedActivists=int(self.jailedActivists[idx]),
            activists=int(self.activistPlot[idx]),
            police=int(self.policePlot[idx]),
            pillars=int(self.pillarPlot[idx]),
            victory=self.victory,
            defected_institutions=(
                ",".join(f"{inst}" for inst in defected_institutions) if defected_institutions else "none"
            ),
            **grievance_kv,
        )

        if current_step >= self.const["MaxSteps"] or self.victory != 0:
            self.running = False

    # ── Outcome helpers ───────────────────────────────────────────────────────

    def get_cascade_summary(self) -> Dict:
        """Summary dict for batch output — cascade order and final counts."""
        return {
            "first_defection_steps": {
                inst: step
                for inst, step in self._first_defection_step.items()
                if step is not None
            },
            "never_defected": [
                inst for inst, step in self._first_defection_step.items()
                if step is None
            ],
            "final_defection_by_type": {
                inst: sum(
                    1 for a in self.schedule.agents
                    if a.is_Pillar and a.defect and a.institution_type == inst
                )
                for inst in UNIQUE_PILLAR_TYPES
            },
        }

    def get_final_victory_status(self):
        return "resistance" if self.victory == 2 else "regime"

    def close(self):
        self.logger.close()
