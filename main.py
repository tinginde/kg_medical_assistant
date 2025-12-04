import kg_model
import rag_engine

def simulate_llm_response(prompt):
    """
    Simulates an LLM response based on the prompt. 
    In a real system, this would call OpenAI/Gemini API.
    """
    # Simple rule-based simulation for the POC
    if "conflicts with managing 'Calorie_Intake'" in prompt:
        return (
            "**Conclusion:** Patient A's slow weight loss is likely driven by their high preference for 'Diet Flexibility'. "
            "This preference creates a conflict with strict 'Calorie Intake' management. "
            "While the patient may be active, the lack of dietary structure is negating the caloric deficit required for weight loss.\n\n"
            "**Recommendation:** Instead of a rigid diet plan (which conflicts with their preference), suggest a 'Volume Eating' approach "
            "or intermittent fasting, which allows for flexibility while naturally limiting caloric window."
        )
    return "Could not determine a specific cause based on the provided context."

def main():
    print("--- Initializing KG-RAG POC for Obesity Management ---")
    
    # 1. Build Graph
    G = kg_model.build_graph()
    print("Knowledge Graph built successfully.")

    # 2. Define Query
    patient_id = "Patient_A"
    query = "What is the connection between Patient A's slow weight loss and their preferences?"
    print(f"\nUser Query: {query}")

    # 3. Retrieve Context (RAG)
    print("\n--- Retrieving Context from Knowledge Graph ---")
    reasoning_text, evidence = rag_engine.retrieve_context(G, patient_id)
    
    for line in reasoning_text:
        print(f"[RAG Info]: {line}")

    # 4. Generate Prompt
    prompt = rag_engine.format_prompt(query, reasoning_text)
    
    # 5. Get LLM Response
    print("\n--- Generating LLM Response ---")
    llm_response = simulate_llm_response(prompt)
    
    # 6. Output Results
    print("\n" + "="*30)
    print("FINAL OUTPUT")
    print("="*30)
    print(llm_response)
    
    print("\n" + "-"*30)
    print("EXPLAINABILITY EVIDENCE (KG Trace)")
    print("-"*30)
    for i, item in enumerate(evidence, 1):
        print(f"{i}. {item['preference']} --[conflicts_with]--> {item['behavior']} --[{item['behavior_relation']}]--> {item['outcome']}")

if __name__ == "__main__":
    main()
