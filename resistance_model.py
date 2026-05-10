"""
resistance_model.py
 
PILLAR INSTITUTION ASSIGNMENT — HAMILTON PROPORTIONAL METHOD
  Previous versions sorted pillars by defectThreshold rank and assigned
  institution types by indexing into a flat PILLAR_TYPE_WEIGHTS list.
 
  This version uses allocate_pillars_proportional(), which implements the
  Hamilton/Largest Remainder method:
    1. Compute each institution type's quota: (weight / total_weight) * NPillar
    2. Floor all quotas → initial allocation
    3. Distribute remaining seats to types with largest fractional remainders,
       with random tie-breaking for equal remainders
    4. Shuffle the result so same-type instances are not placed consecutively
 
  PILLAR_TYPE_WEIGHTS is a Dict[str, float] imported from LLM.schemas,
  keyed by the eight institution types in UNIQUE_PILLAR_TYPES:
    "Educational System"
    "Religious Institution"
    "Business Community"
    "Civil Service"
    "Local Government"
    "Judiciary Institution"
    "Legislative Institution"
    "State-Supported Media"
 
  Institution types with higher weights receive more pillar agent instances,
  reflecting NAVCO historical frequency data: institutions that appear as
  early defectors in more campaigns receive larger weight and are therefore
  more represented in the simulation.
 
DEFECTION RULE — UNCHANGED
  active_protestors_nearby / total_agents_nearby > defectThreshold
  Identical for every pillar. Identical to PoliceAgent.ruleD().
  Institution type is a label only — it changes nothing about the mathematics.
 
CASCADE TRACKING — TWO COMPLEMENTARY STRUCTURES
  _pillar_registry: Dict[int, dict]
    Per-agent defection record, keyed by unique_id, built at init and
    updated by _update_cascade_tracker() on every step.
    Fields: institution, threshold, defection_step (None until defected).
    Provides per-instance resolution: which exact agent defected when.
 
  _first_defection_step: Dict[str, Optional[int]]
    Per-institution-type first defection step, keyed by UNIQUE_PILLAR_TYPES.
    Populated by _update_cascade_tracker() the first time any instance of
    each institution type defects. Used by get_cascade_summary() to
    reconstruct the cascade order across runs without post-processing.
"""


from __future__ import annotations

import json
import os
from typing import List, Dict, Optional
import random
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
from LLM.schemas import PILLAR_TYPE_WEIGHTS, UNIQUE_PILLAR_TYPES


# ── Mesa shape override ───────────────────────────────────────────────────────

def _draw_grid_override(space, space_ax, agent_portrayal):
    """
    Scatter-only grid renderer (superseded by the second definition below).
    Kept for reference. Not active at runtime.
    """
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



def _draw_grid_override(space, space_ax, agent_portrayal):
    """
    Custom Mesa grid renderer supporting both scatter markers and emoji/text labels.
 
    Separates agents into two render passes:
      - Standard agents (Citizens, Police): rendered via ax.scatter with
        marker shapes and colours defined by agent_portrayal().
      - Text/emoji agents (Activists, Pillars): rendered via ax.text,
        distinguished by font size (pillars: size >= 13, activists: size 9).
 
    This function shadows the scatter-only definition above. It is registered
    via override_mesa_shapes(), which patches
    mesa.experimental.components.matplotlib._draw_grid before any
    visualisation is initialised.
    """
    def portray(g):
        """
        Separate all grid agents into scatter-renderable and text-renderable lists.
 
        Returns (standard_plots, text_plots) where standard_plots contains
        dicts for ax.scatter and text_plots contains dicts for ax.text.
        """
        standard_plots = [] # List of dicts for ax.scatter
        text_plots = []     # List of dicts for ax.text

        for i in range(g.width):
            for j in range(g.height):
                content = g._grid[i][j]
                if not content: continue
                if not hasattr(content, "__iter__"): content = [content]
                
                for agent in content:
                    data = agent_portrayal(agent)
                    
                    # IF there is text/emoji, we handle it differently
                    if data.get("text"):
                        text_plots.append({
                            "x": i, "y": j, 
                            "s": data.get("text"), 
                            "color": data.get("color", "black"),
                            "size": data.get("size", 12) # fontsize
                        })
                    else:
                        # Standard scatter logic
                        standard_plots.append({
                            "x": i, "y": j,
                            "marker": data.get("marker", "o"),
                            "c": data.get("color", "gray"),
                            "s": data.get("size", 30)
                        })
        return standard_plots, text_plots

    space_ax.set_xlim(-1, space.width); space_ax.set_ylim(-1, space.height)
    space_ax.tick_params(left=False, right=False, labelleft=False,
                         labelbottom=False, bottom=False)
    space_ax.set_xticks([x + 0.5 for x in range(space.width)])
    space_ax.set_yticks([x + 0.5 for x in range(space.height)])
    space_ax.get_figure().tight_layout()
    space_ax.get_figure().set_figwidth(space_ax.get_figure().get_size_inches()[1])
    space_ax.grid(alpha=0.15)

    std_p, txt_p = portray(space)
    
    # Draw standard agents (Citizens/Police)
    for p in std_p:
        space_ax.scatter(p["x"], p["y"], marker=p["marker"], c=p["c"], s=p["s"])
        
    # Draw Emoji Agents (Pillars/Activists)

    for t in txt_p:
        is_pillar = t["size"] >= 13  # pillars use size 13, activists use 9
        space_ax.text(
            t["x"], t["y"], t["s"],
            color=t["color"],
            ha='center', va='center',
            fontsize=t["size"],
            fontweight='bold',
            fontfamily='DejaVu Sans',
        )
        

def override_mesa_shapes():
    """
    Patch Mesa's internal grid drawing function with _draw_grid_override.
 
    Must be called before JupyterViz or any other Mesa visualisation is
    initialised. Enables mixed scatter (Citizens, Police) and text/emoji
    (Activists, Pillars) rendering on the same grid canvas.
    """
    mesa.experimental.components.matplotlib._draw_grid = _draw_grid_override

# ── NVResistanceModel ─────────────────────────────────────────────────────────
class NVResistanceModel(mesa.Model):
    """
    Agent-based model of nonviolent resistance inspired by Chenoweth & Stephan.
 
    Simulates four interacting agent populations on a toroidal grid:
      - CitizenAgent    — civilian population; may activate into nonviolent protest
      - ActivistAgent   — committed nonviolent leaders; LLM-guided or rule-based
      - PoliceAgent     — regime enforcement; may arrest, kill, or defect
      - PillarAgent     — institutional pillars of the regime; may defect under pressure
 
    The model tests whether LLM-guided activists, reasoning from Gandhian principles
    and compressed episodic memory, produce movement dynamics consistent with
    Chenoweth's empirical findings on nonviolent campaign success.
    """
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
        """
        Initialise the nonviolent resistance simulation.
 
        Sets up the grid, schedule, LLM chat model, data collectors, constants,
        and all agent populations. After agent placement, assigns institution
        types to pillar agents proportionally using the Hamilton method, then
        builds the pillar registry and computes initial defect-level labels.
 
        Parameters
        ----------
        MaxSteps : int
            Maximum simulation steps. The run is forced to outcome 3 (draw)
            if this is reached without a victory condition.
        SizeX, SizeY : int
            Grid dimensions. Total cells = SizeX × SizeY.
        IsTorus : bool
            Whether the grid wraps toroidally (edges connect).
        PctCitizen, PctActivist, PctPolice, PctPillar : float
            Percentage of grid cells occupied by each agent type at init.
        AgentVision : int
            Default Moore-neighbourhood vision radius for all agents.
        ActivistSearchVision : int
            Extended vision radius used by activist agents in strategies 2 and 4.
        StartLegitimacy : float
            Initial government legitimacy [0.0, 1.0]. Decremented on kill
            backfire events via BackfireCoeff.
        PillarProxStrat : int
            Activist movement strategy:
              0 = random
              1 = nearest pillar (normal vision)
              2 = weakest pillar (extended vision)
              3 = LLM-guided (normal vision)
              4 = LLM-guided (extended vision)
        NVSuccessPct : float
            Percentage of pillar agents that must defect for resistance victory
            (stored internally as a fraction, divided by 100).
        dataCollectorType : str
            "singleRunVisualize" — collects live counts for interactive display.
            "batchRun"           — collects summary scalars for batch analysis.
        seed : int or None
            Random seed forwarded to Mesa. Controls agent placement and all
            stochastic rule outcomes for reproducibility.
        init_for_viz : bool
            When True, suppresses heavy init logging (used by the visualisation
            wrapper to avoid polluting the log on non-research launches).
        log_config : LogConfig or None
            Logging and path configuration. Defaults to DEFAULT_LOG_CONFIG.
        """

        super().__init__()

        # ── Logging ───────────────────────────────────────────────────────────
        self._log_config = log_config if log_config is not None else DEFAULT_LOG_CONFIG
        self._log_config.create_all_dirs()
        self._log_config.reset_run_logs()
        self.logger = SimLogger(self._log_config)

        # ── Grid, schedule, LLM ──────────────────────────────────────────────
        self.grid       = mesa.space.SingleGrid(SizeX, SizeY, IsTorus)
        self.schedule   = mesa.time.RandomActivation(self)
        self.chat_model = ChatModel(provider=self._log_config.provider, prompt_type=self._log_config.prompt_type)

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


        for agent_id, type_ in enumerate(TypeLUT):
            agent_class = AGENT_CLASS_MAP[type_]
            agent = agent_class(agent_id, type_, self)
            self.grid.move_to_empty(agent)
            self.schedule.add(agent)

        # ── Sort-then-assign ──────────────────────────────────────────────────
        self._assign_institution_types_proportional(NPillar)

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
    def allocate_pillars_proportional(
        self,
        pillar_weights: Dict[str, float],
        n_total_agents: int,
        random_seed = 42
    ) -> List[str]:
        
        """
        Allocate pillar agents to institution types using the Hamilton method.
 
        Ensures exactly n_total_agents labels are produced while respecting
        the proportional weights of each institution type. Fractional remainder
        seats are distributed to the types with the largest remainders; ties
        are broken by a locally-seeded shuffle that does not affect global
        random state.
 
        Parameters
        ----------
        pillar_weights : Dict[str, float]
            Mapping of institution type → relative weight, as defined in
            PILLAR_TYPE_WEIGHTS. Weights need not sum to 1; they are
            normalised internally.
        n_total_agents : int
            Total number of pillar agents to assign. The returned list has
            exactly this length.
        random_seed : int
            Passed to random.Random() to create an isolated RNG instance
            used only for tie-breaking and the final shuffle. Does not
            mutate the global random state.
 
        Returns
        -------
        List[str]
            Flat list of institution type strings, length == n_total_agents,
            shuffled so same-type instances are not placed consecutively.
 
        Example
        -------
        weights = {
            "Educational System":    2.0,
            "Religious Institution": 1.5,
            "Business Community":    1.0,
            "Civil Service":         1.0,
        }
        result = allocate_pillars_proportional(weights, 6, random_seed=42)
        # Quotas: [2.18, 1.64, 1.09, 1.09] → floors [2, 1, 1, 1] → 1 seat left
        # Largest remainder: Educational System (0.18) gets +1
        # Final counts: {"Educational System": 3, others: 1 each}
        # Returned as shuffled flat list of length 6.
        """
        random_seed = random.Random(random_seed)

        pillars = list(pillar_weights.keys())
        weights = np.array([pillar_weights[p] for p in pillars])
        total_weight = weights.sum()
        
        # Step 1: Calculate quotas
        quotas = (weights / total_weight) * n_total_agents
        
        # Step 2: Initial allocation (floor)
        allocation = np.floor(quotas).astype(int)
        
        # Step 3: Calculate how many seats remain
        remaining = n_total_agents - allocation.sum()
        
        if remaining > 0:
            # Step 4: Get fractional parts (remainders)
            remainders = quotas - allocation
            
            # Create list of (pillar_index, remainder) and shuffle for tie-breaking
            pillar_remainders = list(enumerate(remainders))
            random_seed.shuffle(pillar_remainders)  # Random tie-breaking
            
            # Sort by remainder (descending), ties resolved by shuffle order
            pillar_remainders.sort(key=lambda x: x[1], reverse=True)
            
            # Step 5: Give +1 to top 'remaining' pillars
            for i in range(remaining):
                pillar_idx = pillar_remainders[i][0]
                allocation[pillar_idx] += 1
        
        # Step 6: Build flat list
        result = []
        for pillar, count in zip(pillars, allocation):
            result.extend([pillar] * count)
        
        # Shuffle the list so same pillar types aren't consecutive
        random_seed.shuffle(result)
        
        return result


    # ── Integration into your class ──────────────────────────────────

    def _assign_institution_types_proportional(self, NPillar: int) -> None:
        """
        Assign institution types to all PillarAgents using proportional allocation.
 
        Calls allocate_pillars_proportional() with PILLAR_TYPE_WEIGHTS and
        NPillar, then injects the resulting institution type into each pillar
        agent via _inject_pillar_identity(). Institution types are assigned in
        the shuffled order returned by allocate_pillars_proportional() — there
        is no threshold sorting; thresholds and institution labels are
        independent.
 
        Logs a warning if the number of PillarAgents found in the schedule
        differs from NPillar (can occur if grid placement failed for some agents
        due to a full grid).
 
        Parameters
        ----------
        NPillar : int
            Expected number of pillar agents, used as n_total_agents for the
            Hamilton allocation.
        """
        pillars = [a for a in self.schedule.agents if a.is_Pillar]
        
        if len(pillars) != NPillar:
            self.logger.log_model_init(
                f"WARN: NPillar={NPillar} but {len(pillars)} PillarAgents found."
            )
    
        # Allocate using Hamilton method
        pillar_type_list = self.allocate_pillars_proportional(
            pillar_weights=PILLAR_TYPE_WEIGHTS,
            n_total_agents=NPillar,
        )
        
        # Assign to agents
        for pillar_agent, institution_type in zip(pillars, pillar_type_list):
            pillar_agent._inject_pillar_identity(pillar_agent.defectThreshold, institution_type)

    # ── Spatial helpers ───────────────────────────────────────────────────────
    def get_empty_spaces_in_vision(self, agent, radius):
        """
        Return all empty grid cells within the Moore neighbourhood of agent.
 
        Parameters
        ----------
        agent : NVBaseAgent
            Agent whose position is used as the centre of the search.
        radius : int
            Moore neighbourhood radius to search.
 
        Returns
        -------
        List[Tuple[int, int]]
            List of (x, y) positions that are currently unoccupied.
        """
        neighborhood = self.grid.get_neighborhood(
            pos=agent.pos, moore=True, include_center=False, radius=radius
        )
        return [pos for pos in neighborhood if self.grid.is_cell_empty(pos)]

    # ── Memory retrieval ──────────────────────────────────────────────────────

    def retrieve_agent_cognitive_memory(self, agent_id, step):
        """
        Load an agent's cognitive memory from the NLP memory file for step - 1.
 
        Called by ActivistAgent.step() before ruleM() so the agent has access
        to the previous step's self-reflection and environmental observation
        when making its movement decision.
 
        Returns None silently in two expected cases:
          - step == 0: no previous step exists yet.
          - The agent has no entry in the memory file (e.g. non-LLM strategies
            do not generate movement_decision events, so no memory is written).
 
        Parameters
        ----------
        agent_id : int
            Unique ID of the activist agent requesting memory.
        step : int
            Current simulation step. Memory is read from step - 1.
 
        Returns
        -------
        dict or None
            Cognitive memory dict with structure:
              {
                "cognitive_memory": {
                  "self_reflection":           str,
                  "environmental_observation": str,
                }
              }
            Returns None if the file is absent or the agent has no entry.
        """
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
        """
        Flush, close, and convert a step log into cognitive memories on disk.
 
        Called at the start of each model step to process the PREVIOUS step's
        log before any agent reads memory this step. Full pipeline:
          1. flush_step  — write any buffered log entries to disk
          2. close_step  — finalise and close the step log file
          3. get_activist_neighborhood_memories() — parse log into raw memory
             dicts, applying toroidal distance filtering at vision_radius
          4. save_activist_memories_json() — write raw memory to disk
          5. _convert_raw_to_cognitive_memories() — generate NLP narratives
             and write cognitive memory JSON consumed by
             retrieve_agent_cognitive_memory()
 
        Returns False immediately (without error) if the step log file does
        not exist — this is the expected behaviour for step -1 (before any
        agents have acted).
 
        Parameters
        ----------
        step_num : int
            Step number whose log should be processed. Typically current_step - 1.
        vision_radius : int
            Radius used to filter neighbourhood events in raw memory extraction.
            Should match the activist vision radius for the active strategy.
 
        Returns
        -------
        bool
            True if memories were successfully processed; False if the log
            file for step_num does not exist.
        """
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
        """
        Convert raw memory JSON into NLP narrative memories and write to disk.
 
        Reads the raw memory file for the given step, generates two narrative
        strings per activist agent via the nlp_memory functions, and writes
        the result as the cognitive memory JSON that will be consumed by
        retrieve_agent_cognitive_memory() in the following step.
 
        Output structure per agent_id key:
          {
            "cognitive_memory": {
              "self_reflection":           str,  # generate_gandhian_memory()
              "environmental_observation": str,  # generate_neighborhood_memory()
            }
          }
 
        Also logs the first 120 characters of each self_reflection to the
        narrative log for quick human inspection during a run.
 
        Parameters
        ----------
        step : int
            Step number whose raw memory file should be converted.
            Reads from memory_raw_path(step), writes to memory_nlp_path(step).
        """
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

    # ── Cascade tracker ──────────────────────────────────────────────────────
    def _update_cascade_tracker(self, current_step: int) -> Dict[str, int]:
        """
        Update per-agent and per-institution-type defection records.
 
        For each pillar agent that has defected this step or earlier:
          - On first detection (defection_step is None in _pillar_registry),
            records current_step in _pillar_registry for that agent.
          - If this is also the first instance of its institution type to
            defect, records current_step in _first_defection_step.
          - Emits a DEFECTION model state log event on the step each
            individual agent's defection is first recorded.
 
        Called once per model step after schedule.step().
 
        Parameters
        ----------
        current_step : int
            The step number at which defections are being checked.
 
        Returns
        -------
        Dict[str, int]
            Count of currently defected pillar agents per institution type.
            All UNIQUE_PILLAR_TYPES are present as keys; value is 0 for
            types with no defected instances.
        """
        per_type: Dict[str, int] = {inst: 0 for inst in UNIQUE_PILLAR_TYPES}

        for a in self.schedule.agents:
            if not a.is_Pillar:
                continue
            if a.defect:
                per_type[a.institution_type] += 1
                if self._pillar_registry[a.unique_id]["defection_step"] is None:
                    self._pillar_registry[a.unique_id]["defection_step"] = current_step
                    # Track first defection of this institution type
                    if self._first_defection_step[a.institution_type] is None:
                        self._first_defection_step[a.institution_type] = current_step
                        
                    self.logger.log_model_state(
                        current_step,
                        event="DEFECTION",
                        institution=a.institution_type,
                        threshold=f"{a.defectThreshold:.4f}",
                    )

        return per_type

    # ── Grievance / defect level updates ─────────────────────────────────────

    def update_citizen_grievance_levels(self):
        """
        Classify each citizen's grievance into tertiles and update grievanceLevel.
 
        Computes the 33rd and 66th percentile of current grievance values across
        all living citizen agents, then assigns each citizen:
          "low"    — grievance ≤ 33rd percentile
          "medium" — grievance ≤ 66th percentile
          "high"   — otherwise
 
        grievanceLevel feeds into parse_context_by_direction(), where citizens
        with grievanceLevel == "high" are labelled "suffering" in the directional
        context sent to the LLM activist agent.
 
        Called once per model step after schedule.step().
 
        Returns
        -------
        str
            Space-separated key=value string for inclusion in the model state
            log, e.g.:
              "GrievanceLow=400 GrievanceMedium=380 GrievanceHigh=370
               q33=0.123 q66=0.234"
            Returns an empty string if no citizen agents are alive.
        """
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
        """
        Classify each pillar agent's defection susceptibility into tertiles.
 
        Computes the 33rd and 66th percentile of defectThreshold values across
        all pillar agents, then assigns each pillar a defectLevel label:
          "high"   — defectThreshold ≤ 33rd percentile (defects earliest)
          "medium" — defectThreshold ≤ 66th percentile
          "low"    — otherwise (defects latest or not at all)
 
        Called once at model initialisation. Since defectThreshold is a fixed
        per-agent attribute, defectLevel does not change during a run.
 
        Note: defectLevel is an internal bookkeeping label used for researcher
        inspection only. It is NOT passed to the LLM — activist agents observe
        only defectStatus (defected / stable) via parse_context_by_direction().
        """
        agents = [a for a in self.schedule.agents if a.is_Pillar]
        if not agents:
            return
        thresholds = np.array([a.defectThreshold for a in agents])
        q1, q2     = np.quantile(thresholds, [0.33, 0.66])
        counts     = {"low": 0, "medium": 0, "high": 0}
        for a in agents:
            if a.defectThreshold <= q1:   a.defectLevel = "high";   counts["high"]   += 1
            elif a.defectThreshold <= q2: a.defectLevel = "medium"; counts["medium"] += 1
            else:                         a.defectLevel = "low";    counts["low"]    += 1
        self.logger.log_model_state(
            self.schedule.steps, event="defect_levels",
            high=counts["high"], medium=counts["medium"], low=counts["low"],
            q33=f"{q1:.3f}", q66=f"{q2:.3f}",
        )

    # ── Victory condition ─────────────────────────────────────────────────────
    def check_victory_condition(self):
        """
        Evaluate whether a terminal outcome has been reached this step.
 
        Checks two conditions in priority order:
          1. Resistance victory (code 2): the fraction of defected pillar agents
             exceeds NVSuccessPct. Checked first because pillar collapse is the
             primary Chenoweth success criterion.
          2. Regime victory (code 1): the count of alive nonviolent agents
             (all activists + citizens with activeNV == True) falls to or below
             nNVthreshold. Movement has been suppressed below critical mass.
 
        Returns
        -------
        int
            0 — no terminal condition; simulation continues.
            1 — regime victory: nonviolent resistance suppressed.
            2 — resistance victory: sufficient institutional defections achieved.
        """
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
        """
        Advance the simulation by one step.
 
        Execution order within each step:
          1. process_step_memories(prev_step) — convert the previous step's
             event log into cognitive memories before agents act this step.
          2. schedule.step() — all agents execute their step() in random order.
          3. update_citizen_grievance_levels() — recompute grievance tertiles
             based on updated grievance values after schedule execution.
          4. Time-series arrays updated (citizenPlot, protestingCitizen, etc.)
             using idx = current_step - 1 as the array index.
          5. Max metrics updated (Max_InProtestPct, Max_ActualInProtestPct,
             Max_NVCitizens) — running maximums across the full run.
          6. check_victory_condition() — evaluate terminal conditions.
          7. datacollector.collect() — snapshot model-level reporters.
          8. _update_cascade_tracker() — record new pillar defection events
             in _pillar_registry and _first_defection_step.
          9. log_model_state() — write full model state summary to the log,
             including legitimacy, agent counts, victory code, and grievance
             distribution.
         10. Set self.running = False if MaxSteps is reached or victory != 0,
             signalling Mesa to stop the run.
        """
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
        """
        Return cascade timeline and final defection counts for batch output.
 
        Reads from _first_defection_step (populated per institution type during
        the run) and _pillar_registry (populated per agent) to reconstruct the
        full defection history without additional log parsing.
 
        Returns
        -------
        dict
            Three keys:
 
            "first_defection_steps" : Dict[str, int]
                Institution type → step of first defection. Only institution
                types that defected at least once appear here.
 
            "never_defected" : List[str]
                Institution types with no defections during the run.
 
            "final_defection_by_type" : Dict[str, int]
                Institution type → count of defected instances at end of run.
                All UNIQUE_PILLAR_TYPES are present; 0 for non-defected types.
        """
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
        """
        Return a human-readable string describing the run outcome.
 
        Returns
        -------
        str
            "resistance" — resistance won (victory == 2).
            "regime"     — regime won or draw (victory == 1 or 3).
        """
        return "resistance" if self.victory == 2 else "regime"

    def close(self):
        self.logger.close()
