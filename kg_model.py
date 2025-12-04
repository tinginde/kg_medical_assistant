import networkx as nx

def build_graph():
    """
    Builds the Knowledge Graph for the Obesity Management POC.
    Nodes: Patient, Metric, Preference, Outcome, Behavior
    Edges: has_metric, has_preference, influenced_by, mitigates, conflicts_with
    """
    G = nx.DiGraph()

    # --- Shared Nodes (Metrics and Behaviors) ---
    # Metrics (COA)
    G.add_node("Weight_Metric", type="Metric", description="Body Weight")
    G.add_node("BMI", type="Metric", description="Body Mass Index")
    G.add_node("Stress_Score", type="Metric", description="Self-reported stress level")

    # Behaviors (DHT)
    G.add_node("Calorie_Intake", type="Behavior", description="Daily caloric consumption")
    G.add_node("Activity_Level", type="Behavior", description="Daily step count")
    G.add_node("Sleep_Quality", type="Behavior", description="Sleep duration and quality")

    # --- Patient A: Diet Flexibility Conflict ---
    G.add_node("Patient_A", type="Patient", description="Patient with slow weight loss")
    G.add_node("Diet_Flexibility_Preference", type="Preference", description="Preference for flexible diet choices")
    G.add_node("Rate_of_Loss_Preference_A", type="Preference", description="Preference for rapid weight loss")
    G.add_node("Slow_Weight_Loss_A", type="Outcome", description="Weight loss < 0.5kg/week")
    G.add_node("High_Stress_A", type="Outcome", description="Stress score > 7/10")

    G.add_edge("Patient_A", "Slow_Weight_Loss_A", relation="experiences")
    G.add_edge("Patient_A", "High_Stress_A", relation="experiences")
    G.add_edge("Patient_A", "Diet_Flexibility_Preference", relation="has_preference")
    G.add_edge("Patient_A", "Rate_of_Loss_Preference_A", relation="has_preference")
    
    # Conflict: Diet Flexibility vs Calorie Control
    G.add_edge("Diet_Flexibility_Preference", "Calorie_Intake", relation="conflicts_with", 
               reason="Flexible dieting often leads to underestimation of calories")
    G.add_edge("Calorie_Intake", "Slow_Weight_Loss_A", relation="influences", effect="positive_correlation")
    G.add_edge("Activity_Level", "Slow_Weight_Loss_A", relation="mitigates")
    G.add_edge("High_Stress_A", "Calorie_Intake", relation="increases")

    # --- Patient B: High Activity but Stress Eating ---
    G.add_node("Patient_B", type="Patient", description="High activity but poor weight loss due to stress eating")
    G.add_node("Exercise_Preference", type="Preference", description="Strong preference for exercise-based weight loss")
    G.add_node("Minimal_Diet_Preference", type="Preference", description="Prefers not to restrict diet")
    G.add_node("Slow_Weight_Loss_B", type="Outcome", description="Weight loss < 0.5kg/week despite high activity")
    G.add_node("High_Stress_B", type="Outcome", description="Work-related stress")

    G.add_edge("Patient_B", "Slow_Weight_Loss_B", relation="experiences")
    G.add_edge("Patient_B", "High_Stress_B", relation="experiences")
    G.add_edge("Patient_B", "Exercise_Preference", relation="has_preference")
    G.add_edge("Patient_B", "Minimal_Diet_Preference", relation="has_preference")
    
    # Conflict: Exercise focus neglects calorie intake
    G.add_edge("Minimal_Diet_Preference", "Calorie_Intake", relation="conflicts_with",
               reason="Avoiding diet control leads to compensatory eating after exercise")
    G.add_edge("High_Stress_B", "Calorie_Intake", relation="increases")
    G.add_edge("Calorie_Intake", "Slow_Weight_Loss_B", relation="influences", effect="positive_correlation")
    G.add_edge("Activity_Level", "Slow_Weight_Loss_B", relation="mitigates")

    # --- Patient C: Good Adherence but Plateaued ---
    G.add_node("Patient_C", type="Patient", description="Good adherence but weight loss plateau")
    G.add_node("Strict_Adherence_Preference", type="Preference", description="Prefers strict, consistent routine")
    G.add_node("Weight_Plateau", type="Outcome", description="No weight loss for 4+ weeks")
    G.add_node("Good_Adherence", type="Outcome", description="95%+ adherence to plan")
    G.add_node("Metabolic_Adaptation", type="Behavior", description="Body adapted to current calorie deficit")

    G.add_edge("Patient_C", "Weight_Plateau", relation="experiences")
    G.add_edge("Patient_C", "Good_Adherence", relation="experiences")
    G.add_edge("Patient_C", "Strict_Adherence_Preference", relation="has_preference")
    
    # Conflict: Strict adherence to same routine causes adaptation
    G.add_edge("Strict_Adherence_Preference", "Metabolic_Adaptation", relation="conflicts_with",
               reason="Lack of variation allows metabolic adaptation to occur")
    G.add_edge("Metabolic_Adaptation", "Weight_Plateau", relation="causes")
    G.add_edge("Good_Adherence", "Weight_Plateau", relation="insufficient_for_progress")

    return G

def get_patient_scenarios():
    """Returns a dictionary of patient scenarios for testing."""
    return {
        "Patient_A": {
            "query": "Why is Patient A experiencing slow weight loss despite being moderately active?",
            "description": "Conflict between diet flexibility preference and calorie management"
        },
        "Patient_B": {
            "query": "Why is Patient B not losing weight despite high exercise levels?",
            "description": "Stress eating compensates for exercise, minimal diet awareness"
        },
        "Patient_C": {
            "query": "Why has Patient C plateaued despite excellent adherence?",
            "description": "Metabolic adaptation due to lack of variation in routine"
        }
    }

if __name__ == "__main__":
    G = build_graph()
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print(f"\nAvailable patient scenarios: {list(get_patient_scenarios().keys())}")
