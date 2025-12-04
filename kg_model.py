import networkx as nx

def build_graph():
    """
    Builds the Knowledge Graph for the Obesity Management POC.
    Nodes: Patient, Metric, Preference, Outcome, Behavior
    Edges: has_metric, has_preference, influenced_by, mitigates, conflicts_with
    """
    G = nx.DiGraph()

    # --- Nodes ---
    # Patient
    G.add_node("Patient_A", type="Patient", description="Patient with slow weight loss")

    # Metrics (COA)
    G.add_node("Weight_Metric", type="Metric", description="Body Weight")
    G.add_node("BMI", type="Metric", description="Body Mass Index")
    G.add_node("Stress_Score", type="Metric", description="Self-reported stress level")

    # Behaviors (DHT)
    G.add_node("Calorie_Intake", type="Behavior", description="Daily caloric consumption")
    G.add_node("Activity_Level", type="Behavior", description="Daily step count")

    # Preferences (PPI)
    G.add_node("Diet_Flexibility_Preference", type="Preference", description="Preference for flexible diet choices")
    G.add_node("Rate_of_Loss_Preference", type="Preference", description="Preference for rapid weight loss")

    # Outcomes
    G.add_node("Slow_Weight_Loss", type="Outcome", description="Weight loss < 0.5kg/week")
    G.add_node("High_Stress", type="Outcome", description="Stress score > 7/10")

    # --- Edges (Relationships) ---
    
    # Patient Attributes
    G.add_edge("Patient_A", "Slow_Weight_Loss", relation="experiences")
    G.add_edge("Patient_A", "High_Stress", relation="experiences")
    G.add_edge("Patient_A", "Diet_Flexibility_Preference", relation="has_preference")
    G.add_edge("Patient_A", "Rate_of_Loss_Preference", relation="has_preference")

    # Causal / Influence Relationships
    # High Calorie Intake leads to Slow Weight Loss (or prevents fast weight loss)
    G.add_edge("Calorie_Intake", "Slow_Weight_Loss", relation="influences", effect="positive_correlation")
    
    # Activity Level helps mitigate Slow Weight Loss
    G.add_edge("Activity_Level", "Slow_Weight_Loss", relation="mitigates")

    # Stress influences behaviors
    G.add_edge("High_Stress", "Calorie_Intake", relation="increases")
    G.add_edge("High_Stress", "Activity_Level", relation="decreases")

    # Conflicts (The Core Logic)
    # Diet Flexibility Preference often conflicts with strict Calorie Intake control
    G.add_edge("Diet_Flexibility_Preference", "Calorie_Intake", relation="conflicts_with", 
               reason="Flexible dieting often leads to underestimation of calories")

    return G

if __name__ == "__main__":
    G = build_graph()
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
