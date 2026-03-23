<context>
Current neighborhood state:
{{ACTIVIST_LOCAL_CONTEXT}}
</context>


<goal>
As, you prepare for your move, carefully analyze what you see around you. Keeping in view Gandhi strategies, identify the position that will best contribute towards strengthening the non-violent resistance in the present situation.
</goal>


<output_rules>
Respond using exactly the structure below.
Every field is required. Do not skip or add fields.
Your chosen_position must be a coordinate that appears in can_move_to. No other value is valid.


<output_rules>

Every field is required. Do not skip or add fields. Your chosen_position must be a coordinate that appears in can_move_to. No other value is valid. Respond with this exact JSON structure and nothing else:
{
  "decision": {
    "chosen_position": [x, y],
    "target_agent_group": "citizens | pillars | police | activists",
    "target_pillar_institution": "If and only if, the target_agent_group is 'pillars', mention the target institution type, visible in your neighborhood. Write "N/A" otherwise.",
    "primary_attribute": "Gandhian attribute",
    "call_to_action_type": "sit_in | march | boycott | strike | civil_disobedience | petition | hunger_strike"
  }
}

</output_rules>


