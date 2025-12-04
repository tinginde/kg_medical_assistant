import networkx as nx
from data_loader import PatientDataLoader

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

def build_graph_from_csv(csv_path="mok.csv"):
    """
    Builds a Knowledge Graph dynamically from CSV patient data.
    Returns both the graph and the patient data.
    """
    # Load patient data
    loader = PatientDataLoader(csv_path)
    patients = loader.process_data()
    
    G = nx.DiGraph()
    
    # Create shared behavior and metric nodes
    G.add_node("Activity_Level_CSV", type="Behavior", description="Daily step count (from CSV)")
    G.add_node("Calorie_Intake_CSV", type="Behavior", description="Daily caloric intake (from CSV)")
    G.add_node("Weight_Change_Metric", type="Metric", description="Weight change over time")
    G.add_node("HbA1c_Metric", type="Metric", description="Glycated hemoglobin")
    
    # Process each patient
    for patient in patients:
        patient_id = patient['id']
        
        # Create patient node
        G.add_node(patient_id, type="Patient", 
                  description=f"CSV Patient: {patient['weight_change_category']} weight change",
                  data=patient)
        
        # Create preference nodes (unique per patient)
        diet_flex_pref = f"{patient_id}_DietFlexPref"
        weight_loss_pref = f"{patient_id}_WeightLossPref"
        
        G.add_node(diet_flex_pref, type="Preference",
                  description=f"Dietary flexibility preference: {patient['diet_flexibility_label']}",
                  score=patient['diet_flexibility_score'])
        
        G.add_node(weight_loss_pref, type="Preference",
                  description=f"Weight loss rate preference: {patient['weight_loss_pref_label']}",
                  score=patient['weight_loss_pref_score'])
        
        # Create outcome node
        outcome_node = f"{patient_id}_Outcome"
        G.add_node(outcome_node, type="Outcome",
                  description=f"{patient['weight_change_category']} weight change: {patient['weight_change_value']} kg",
                  weight_change=patient['weight_change_value'],
                  category=patient['weight_change_category'])
        
        # Connect patient to preferences and outcome
        G.add_edge(patient_id, diet_flex_pref, relation="has_preference")
        G.add_edge(patient_id, weight_loss_pref, relation="has_preference")
        G.add_edge(patient_id, outcome_node, relation="experiences")
        
        # Create edges based on data patterns
        # Rule 1: High diet flexibility + High caloric intake → Conflict
        if patient['diet_flexibility_class'] == 'High' and patient['caloric_intake_class'] == 'High':
            G.add_edge(diet_flex_pref, "Calorie_Intake_CSV", relation="conflicts_with",
                      reason=f"High flexibility ({patient['diet_flexibility_score']}/10) correlates with high intake ({patient['caloric_intake']} cal)")
            G.add_edge("Calorie_Intake_CSV", outcome_node, relation="influences",
                      effect="negative" if patient['weight_change_value'] < -1 else "neutral")
        
        # Rule 2: High activity + Low intake → Positive outcome
        if patient['daily_steps_class'] == 'High' and patient['caloric_intake_class'] == 'Low':
            G.add_edge("Activity_Level_CSV", outcome_node, relation="strongly_supports",
                      reason=f"High activity ({patient['daily_steps']} steps) with controlled intake")
        
        # Rule 3: Low activity + High intake → Poor outcome
        if patient['daily_steps_class'] == 'Low' and patient['caloric_intake_class'] == 'High':
            G.add_edge("Activity_Level_CSV", outcome_node, relation="insufficient",
                      reason=f"Low activity ({patient['daily_steps']} steps) cannot offset high intake")
            G.add_edge("Calorie_Intake_CSV", outcome_node, relation="dominates",
                      reason=f"High intake ({patient['caloric_intake']} cal) dominates outcome")
        
        # Rule 4: Exercise cannot compensate (high activity but still slow/increase)
        if patient['daily_steps_class'] == 'High' and patient['weight_change_category'] in ['Slow', 'Increase']:
            G.add_edge("Calorie_Intake_CSV", outcome_node, relation="compensates_for",
                      reason="High caloric intake negates exercise benefits")
    
    return G, patients

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
    # Test hardcoded scenarios
    G = build_graph()
    print(f"Hardcoded graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print(f"\nAvailable patient scenarios: {list(get_patient_scenarios().keys())}")
    
    # Test CSV loading
    print("\n" + "="*60)
    print("Testing CSV loading...")
    G_csv, patients = build_graph_from_csv("mok.csv")
    print(f"CSV graph built with {G_csv.number_of_nodes()} nodes and {G_csv.number_of_edges()} edges.")
    print(f"Loaded {len(patients)} patients from CSV.")
