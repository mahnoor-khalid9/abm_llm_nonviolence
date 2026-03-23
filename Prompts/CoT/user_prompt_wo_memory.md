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