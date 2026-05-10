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
As, you prepare to move, naturally weigh what recent experiences have taught you against what you now see around you. Keeping in view Gandhi's strategies, identify the position that will best contribute towards strengthening the non-violent resistance in the present situation.
</goal>


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
    "target_pillar_institution": "Educational System | Trade Unions | Religious Institutions | Professional Associations | Independent Media | Business Community | Civil Service | Local Government | Parliament | Supreme Court | State Media | N/A",
    "primary_attribute": "Gandhian attribute",
    "call_to_action_type": "sit_in | march | boycott | strike | civil_disobedience | petition | hunger_strike"
  }
}
</output_rules>
