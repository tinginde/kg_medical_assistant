import networkx as nx

def retrieve_context(G, patient_id, query_intent="explain_slow_weight_loss"):
    """
    Retrieves a subgraph or reasoning chain relevant to the patient's condition.
    """
    evidence = []
    reasoning_text = []

    # 1. Identify Patient's specific outcomes
    outcomes = [n for n in G.successors(patient_id) if G.nodes[n].get('type') == 'Outcome']
    
    # 2. Identify Patient's Preferences
    preferences = [n for n in G.successors(patient_id) if G.nodes[n].get('type') == 'Preference']

    reasoning_text.append(f"Analyzing {patient_id} who is experiencing: {', '.join(outcomes)}.")

    # 3. Find conflicts between Preferences and Behaviors that influence Outcomes
    for pref in preferences:
        # Check if this preference conflicts with anything
        for neighbor in G.successors(pref):
            edge_data = G.get_edge_data(pref, neighbor)
            if edge_data.get('relation') == 'conflicts_with':
                conflict_node = neighbor # e.g., Calorie_Intake
                
                # Check if this conflict node influences the negative outcome
                for outcome in outcomes:
                    if G.has_edge(conflict_node, outcome):
                        influence_data = G.get_edge_data(conflict_node, outcome)
                        
                        # Construct the evidence chain
                        chain = {
                            "preference": pref,
                            "conflict_relation": "conflicts_with",
                            "behavior": conflict_node,
                            "behavior_relation": influence_data.get('relation'),
                            "outcome": outcome
                        }
                        evidence.append(chain)
                        
                        # Construct natural language explanation
                        explanation = (
                            f"Patient has preference '{pref}'. "
                            f"This conflicts with managing '{conflict_node}' "
                            f"({edge_data.get('reason', 'direct conflict')}). "
                            f"Unmanaged '{conflict_node}' {influence_data.get('relation')} '{outcome}'."
                        )
                        reasoning_text.append(explanation)

    return reasoning_text, evidence

def format_prompt(query, reasoning_text):
    context_str = "\n".join(reasoning_text)
    prompt = f"""
You are a clinical decision support AI. Use the following Knowledge Graph context to answer the user's query.

Context:
{context_str}

Query: {query}

Provide a concise clinical explanation and recommendation.
"""
    return prompt
