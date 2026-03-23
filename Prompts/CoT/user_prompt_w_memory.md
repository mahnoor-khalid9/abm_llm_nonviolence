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


<goal>
Keeping in view your previous action and the changes observed in the
neighborhood, carefully analyze your surroundings from all directions.
Identify the position that will best contribute toward strengthening
non-violent resistance in the present situation guided by Gandhian strategy.
</goal>



<output_rules>
Respond using exactly the structure below. 
Every section is required. Do not skip or merge sections.
Your final chosen_position must be a coordinate that appears in the
can_move_to lists in the context. No other value is valid.

<reasoning>

  <step_1_memory>
  What did your last action achieve?
  Did anything change around you — citizens, activists, structures, police?
  Are there fellow activists visible now that weren't before?
  One sentence conclusion.
  [Write your answer here]
  </step_1_memory>

  <step_2_direction_survey>
  Go through every direction listed in the context one by one.
  For each direction write:
    - DIRECTION NAME
    - Can move there: YES / NO (NO if can_move_to is empty)
    - Agents present: [list types and states]
    - What your presence there would achieve: [one sentence]
  [Write your survey here]
  </step_2_direction_survey>

  <step_3_intersection_opportunity>
  Are there positions that sit at the intersection of two or more zones,
  influencing agents from multiple directions simultaneously?
  Identify any such positions and the combined effect they offer.
  If none exist, state that clearly.
  [Write your answer here]
  </step_3_intersection_opportunity>

  <step_4_top_two_candidates>
  Name your top two candidate positions based on steps 2 and 3.
  For each candidate write:
    - Exact position (coordinates from can_move_to)
    - Agent groups within influence of this position
    - What you would target and why
    - What you would lose by choosing this over the other
  [Write your two candidates here]
  </step_4_top_two_candidates>

  <step_5_elimination>
  Before comparing candidates, assess where you are
  in the Gandhian sequence:
    - Are fellow activists visible? Have you joined them?
    - Are you already present among suffering communities?
    - Is the movement visible enough for structures to witness?

  Then compare your two candidates:
    - Which one moves you closer to the next stage?
    - What does your reflection tell you about where
      your presence is actually making a difference?

  Name the position you are eliminating and state exactly why.
  [Write your elimination here]
  </step_5_elimination>

  <step_6_commit>
  State your final chosen position as a coordinate pair.
  It must match a position named in step_4 and must appear in can_move_to.
  Chosen position: [coordinates]
  </step_6_commit>

</reasoning>

<decision>
  <chosen_position>[Must exactly match step_6_commit]</chosen_position>
  <target_group>[citizens / pillars / police / activists]</target_group>
  <strategy>[Non-violent strategy name]</strategy>
  <primary_attribute>[Gandhian attribute]</primary_attribute>
  <secondary_attribute>[Gandhian attribute]</secondary_attribute>
  <citizen_call_to_action>
  A direct, non-violent appeal addressed to citizens in the current neighborhood.
  Must reference their observed state and invite a specific peaceful action.
  </citizen_call_to_action>
  <explanation>
  Reference your step_5_elimination directly.
  Why this position over the one you eliminated,
  in light of your reflection and current neighborhood signals.
  </explanation>
</decision>

</output_rules>

