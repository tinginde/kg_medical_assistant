import networkx as nx

def retrieve_context(G, patient_id, query_intent="explain_outcome"):
    """
    Retrieves a subgraph or reasoning chain relevant to the patient's condition.
    Enhanced to handle multiple patient scenarios.
    """
    evidence = []
    reasoning_text = []

    if patient_id not in G.nodes():
        return [f"Patient {patient_id} not found in knowledge graph."], []

    # 1. Identify Patient's specific outcomes
    outcomes = [n for n in G.successors(patient_id) if G.nodes[n].get('type') == 'Outcome']
    
    # 2. Identify Patient's Preferences
    preferences = [n for n in G.successors(patient_id) if G.nodes[n].get('type') == 'Preference']

    reasoning_text.append(f"Analyzing {patient_id} who is experiencing: {', '.join(outcomes)}.")

    # 3. Find conflicts between Preferences and Behaviors/Outcomes
    for pref in preferences:
        # Check if this preference conflicts with anything
        for neighbor in G.successors(pref):
            edge_data = G.get_edge_data(pref, neighbor)
            relation = edge_data.get('relation')
            
            if relation == 'conflicts_with':
                conflict_node = neighbor  # e.g., Calorie_Intake or Metabolic_Adaptation
                
                # Check if this conflict node influences/causes any outcome
                for outcome in outcomes:
                    # Direct path: preference -> conflicts_with -> behavior -> influences -> outcome
                    if G.has_edge(conflict_node, outcome):
                        influence_data = G.get_edge_data(conflict_node, outcome)
                        
                        chain = {
                            "preference": pref,
                            "conflict_relation": "conflicts_with",
                            "behavior": conflict_node,
                            "behavior_relation": influence_data.get('relation'),
                            "outcome": outcome
                        }
                        evidence.append(chain)
                        
                        explanation = (
                            f"Patient has preference '{pref}'. "
                            f"This conflicts with managing '{conflict_node}' "
                            f"({edge_data.get('reason', 'direct conflict')}). "
                            f"Unmanaged '{conflict_node}' {influence_data.get('relation')} '{outcome}'."
                        )
                        reasoning_text.append(explanation)
    
    # 4. If no conflicts found, look for other causal relationships
    if not evidence:
        reasoning_text.append("No direct preference-behavior conflicts detected. Analyzing other factors...")
        
        # Look for behaviors that influence outcomes
        for outcome in outcomes:
            # Find all predecessors that influence this outcome
            for pred in G.predecessors(outcome):
                edge_data = G.get_edge_data(pred, outcome)
                if G.nodes[pred].get('type') in ['Behavior', 'Outcome']:
                    explanation = f"'{pred}' {edge_data.get('relation', 'affects')} '{outcome}'."
                    reasoning_text.append(explanation)

    return reasoning_text, evidence

def format_prompt(query, reasoning_text):
    """Format a prompt for LLM with knowledge graph context."""
    context_str = "\n".join(reasoning_text)
    prompt = f"""You are a clinical decision support AI. Use the following Knowledge Graph context to answer the user's query.

Context:
{context_str}

Query: {query}

Provide a concise clinical explanation and recommendation based on the context provided."""
    return prompt
