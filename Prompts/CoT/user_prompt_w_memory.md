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
Based on your memory and neighborhood, decide where to move and what action
to take. Consider what you know about how regimes fall and how nonviolent
movements succeed. Your decision should reflect your Gandhian principles.
</task>


<output_rules>
Every field is required. Do not skip or add fields. Your chosen_position must be
a coordinate that appears in can_move_to. No other value is valid. Respond with
this exact JSON structure and nothing else.

{
  "reasoning": {
    "observation": "What do you see in your neighborhood that matters most? Explain in 1 sentence.",
    "strategic_intent": "Why are you choosing this target and position? Explain in 1 sentence."
  },
  "decision": {
    "chosen_position": [x, y],
    "target_agent_group": "citizens | pillars | police | activists",
    "target_pillar_institution": "Institution name or N/A",
    "primary_attribute": "Gandhian attribute",
    "non_violent_action_method": "protest_and_persuasion | non-cooperation | intervention"
  }
}
</output_rules>
