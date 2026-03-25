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

Respond using ONLY the structure below. Do NOT generate full sentences. Use ONLY 1–3 word phrases or predefined labels.

<reasoning>
  <memory>
    <last_influence>
      [none | citizens_mobilized | activists_clustered | pillar_pressured | police_triggered]
    </last_influence>

    <change_detected>
      [none | more_support | more_risk | mixed]
    </change_detected>
  </memory>

  <situation>
    <primary_opportunity>
      [suffering_citizens | activist_cluster | pillar | police]
    </primary_opportunity>

    <secondary_opportunity>
      [suffering_citizens | activist_cluster | pillar | police ]
    </secondary_opportunity>

    <risk_level>
      [low | medium | high]
    </risk_level>
  </situation>

</reasoning>

<decision>
  <chosen_position>[x, y] — must appear in can_move_to</chosen_position>

  <target_agent_group>
    [citizens | activists | pillars | police]
  </target_agent_group>
  <target_pillar_institution>
    [Educational System | Trade Unions | Religious Institutions | 
    Professional Associations | Independent Media | Business Community |
    Civil Service | Local Government | Parliament | Supreme Court | 
    State Media | N/A]
  </target_pillar_institution>
  <primary_attribute>[Gandhian attribute]</primary_attribute>
  <call_to_action_type>
    [sit_in | march | boycott | strike | civil_disobedience | petition | hunger_strike]
  </call_to_action_type>

</decision>

</output_rules>
