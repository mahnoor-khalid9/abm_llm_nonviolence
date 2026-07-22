<context>
Current neighborhood state:
{{ACTIVIST_LOCAL_CONTEXT}}
</context>


<goal>
As, you prepare for your move, carefully analyze what you see around you. Keeping in view Gandhi strategies, identify the position that will best contribute towards strengthening the non-violent resistance in the present situation.
</goal>


<output_rules>
Respond using exactly the structure below.
Every section is required. Do not skip or merge sections.
Your final chosen_position must be a coordinate that appears in the
can_move_to lists in the context. No other value is valid.

<reasoning>

  <step_1_direction_survey>
  Go through every direction listed in the context one by one.
  For each direction write:
    - DIRECTION NAME
    - Can move there: YES / NO (NO if can_move_to is empty)
    - Agents present: [list types and states]
    - What your presence there would achieve: [one sentence]
  [Write your survey here]
  </step_1_direction_survey>

  <step_2_intersection_opportunity>
  Are there positions that sit at the intersection of two or more zones,
  influencing agents from multiple directions simultaneously?
  Identify any such positions and the combined effect they offer.
  If none exist, state that clearly.
  [Write your answer here]
  </step_2_intersection_opportunity>

  <step_3_top_two_candidates>
  Name your top two candidate positions based on steps 1 and 2.
  For each candidate write:
    - Exact position (coordinates from can_move_to)
    - Agent groups within influence of this position
    - What you would target and why
    - What you would lose by choosing this over the other
  [Write your two candidates here]
  </step_3_top_two_candidates>

  <step_4_elimination>
  Assess where the movement currently stands in the Gandhian sequence:
    - Are fellow activists visible? Have you joined them?
    - Are suffering communities present and reachable?
    - Is the movement visible enough for witnesses to react?

  Then compare your two candidates:
    - Which one moves the movement closer to the next stage?

  Name the position you are eliminating and state exactly why.
  [Write your elimination here]
  </step_4_elimination>

  <step_5_commit>
  State your final chosen position as a coordinate pair.
  It must match a position named in step_3 and must appear in can_move_to.
  Chosen position: [coordinates]
  </step_5_commit>

</reasoning>

<decision>
  <chosen_position>[Must exactly match step_5_commit]</chosen_position>
  <target_group>[citizens / pillars / police / activists]</target_group>
  <strategy>[Non-violent strategy name]</strategy>
  <primary_attribute>[Gandhian attribute]</primary_attribute>
  <secondary_attribute>[Gandhian attribute]</secondary_attribute>
  <explanation>
  Reference your step_4_elimination directly.
  Why this position over the one you eliminated,
  in light of the Gandhian sequence and current neighborhood signals.
  </explanation>
</decision>

</output_rules>







<memory>
Reasoning behind your previous movement:
{{SELF_REFLECTION}}

Recent events observed in the neighborhood:
{{RECENT_EVENTS}}
</memory>


<context>
Current neighborhood state:
{{ACTIVIST_LOCAL_CONTEXT}}
</context>


<task>
Based on your memory and neighborhood, decide your next action. Your decisions must follow these principles:
  - Your decisions should clearly reflect your personality traits.
  - Choose `method of non-violent action` consistent with visible neighborhood conditions
</task>



<output_rules>
RULES — READ FIRST:
1. Respond with valid JSON only. No text before or after.
2. chosen_position must be copied exactly from a can_move_to list in the context.
3. step_2_survey MUST always contain ALL 9 directions: north, south, east, west,
   adjacent, northeast, northwest, southeast, southwest.
4. moves must use JSON array syntax [[x,y], [x,y]] — NOT Python tuples (x,y).
5. For pillars: list only institution names whose status is "stable". Null if none visible.
6. All integer fields must be integers, not strings.
7. best_move and chosen_position must be identical.

{
  "reasoning": {
    "step_1_memory": {
      "last_influence": "none | citizens_mobilized | activists_clustered | pillar_pressured | police_triggered",
      "change_detected": "none | more_support | more_risk | mixed",
      "coordination": "isolated | supported"
    },

    "step_2_survey": {
      "<direction_from_context>": {
        "moves": [[x,y]] or null,
        "suffering": 0,
        "activists": 0,
        "pillars": ["Institution Names"] or null
      }
    },

    "step_3_position": {
      "best_move": [x, y],
      "target_group": "citizens | activists | pillars | police",
      "pillar_required": true
    }
  },

  "decision": {
    "chosen_position": [x, y],
    "target_agent_group": "citizens | activists | pillars | police",
    "target_pillar_institution": "Institution name or N/A",
    "primary_attribute": "Gandhian attribute",
    "non_violent_action_method": "protest_and_persuasion | non-cooperation | intervention",
  }
}
</output_rules>
