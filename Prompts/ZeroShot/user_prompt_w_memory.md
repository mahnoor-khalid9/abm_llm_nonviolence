<memory>
Reasoning behind your previous movement:
{{SELF_REFLECTION}}

Recent events observed in your neighborhood:
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

Every field is required. Do not skip or add fields. Your chosen_position must be a coordinate that appears in can_move_to. No other value is valid. Respond with this exact JSON structure and nothing else:
{
  "decision": {
    "chosen_position": [x, y],
    "target_agent_group": "citizens | pillars | police | activists",
    "target_pillar_institution": "Institution name or N/A",
    "primary_attribute": "Gandhian attribute",
    "non_violent_action_method": "protest_and_persuasion | non-cooperation | intervention",
  },
}

</output_rules>
