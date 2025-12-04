import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

import kg_model
import rag_engine
from visualize import visualize_reasoning_chain

# Load environment variables
load_dotenv()

def get_llm_response(prompt):
    """
    Attempts to use a real LLM API if configured, otherwise falls back to simulation.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    # Try OpenAI
    if openai_key:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=openai_key)
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            print(f"[Warning] OpenAI API failed: {e}. Falling back to simulation.")
    
    # Try Google Gemini
    if google_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, google_api_key=google_key)
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            print(f"[Warning] Google API failed: {e}. Falling back to simulation.")
    
    # Fallback to simulation
    return simulate_llm_response(prompt)

def simulate_llm_response(prompt):
    """
    Simulates an LLM response based on the prompt. 
    In a real system, this would call OpenAI/Gemini API.
    """
    # Enhanced simulation with pattern matching
    if "Diet_Flexibility_Preference" in prompt and "Calorie_Intake" in prompt:
        return (
            "**Conclusion:** Patient A's slow weight loss is likely driven by their high preference for 'Diet Flexibility'. "
            "This preference creates a conflict with strict 'Calorie Intake' management. "
            "While the patient may be active, the lack of dietary structure is negating the caloric deficit required for weight loss.\n\n"
            "**Recommendation:** Instead of a rigid diet plan (which conflicts with their preference), suggest a 'Volume Eating' approach "
            "or intermittent fasting, which allows for flexibility while naturally limiting caloric window."
        )
    elif "Minimal_Diet_Preference" in prompt and "Exercise_Preference" in prompt:
        return (
            "**Conclusion:** Patient B is experiencing the classic 'exercise compensation' phenomenon. "
            "Their strong preference for exercise-based weight loss, combined with minimal diet awareness, "
            "leads to compensatory eating that negates the caloric expenditure from exercise. "
            "Stress eating further exacerbates this issue.\n\n"
            "**Recommendation:** Implement mindful eating practices and stress management techniques. "
            "Use a food diary (not restrictive counting) to increase awareness of eating patterns, especially post-workout."
        )
    elif "Metabolic_Adaptation" in prompt or "Weight_Plateau" in prompt:
        return (
            "**Conclusion:** Patient C's plateau is a result of metabolic adaptation. "
            "Their excellent adherence to a consistent routine has allowed their body to adapt, "
            "reducing energy expenditure and halting weight loss progress.\n\n"
            "**Recommendation:** Implement a 'refeed' strategy (1-2 days of maintenance calories weekly) "
            "or introduce structured variation in training (periodization). This disrupts adaptation while "
            "respecting their preference for structure."
        )
    
    return "Based on the provided context, further analysis is needed to determine the root cause."

def process_patient_query(G, patient_id, query):
    """Process a single patient query through the KG-RAG pipeline."""
    print(f"\n{'='*80}")
    print(f"PATIENT: {patient_id}")
    print(f"QUERY: {query}")
    print(f"{'='*80}")
    
    # 1. Retrieve Context (RAG)
    print("\n--- Retrieving Context from Knowledge Graph ---")
    reasoning_text, evidence = rag_engine.retrieve_context(G, patient_id)
    
    for line in reasoning_text:
        print(f"[RAG Info]: {line}")
    
    # 2. Generate Prompt using LangChain PromptTemplate
    template = """You are a clinical decision support AI specialized in obesity management for the UNIFIED project.

Knowledge Graph Context:
{context}

Patient Query: {query}

Based on the context above, provide:
1. A clear clinical explanation of the patient's situation
2. Specific, actionable recommendations that respect their preferences

Response:"""
    
    prompt_template = PromptTemplate(
        input_variables=["context", "query"],
        template=template
    )
    
    context_str = "\n".join(reasoning_text)
    formatted_prompt = prompt_template.format(context=context_str, query=query)
    
    # 3. Get LLM Response
    print("\n--- Generating LLM Response ---")
    llm_response = get_llm_response(formatted_prompt)
    
    # 4. Output Results
    print(f"\n{'='*80}")
    print("FINAL OUTPUT")
    print(f"{'='*80}")
    print(llm_response)
    
    print(f"\n{'-'*80}")
    print("EXPLAINABILITY EVIDENCE (KG Trace)")
    print(f"{'-'*80}")
    if evidence:
        for i, item in enumerate(evidence, 1):
            print(f"{i}. {item['preference']} --[{item['conflict_relation']}]--> "
                  f"{item['behavior']} --[{item['behavior_relation']}]--> {item['outcome']}")
    else:
        print("No direct conflict chains found. See context above for causal relationships.")
    
    # 5. Generate visualization
    if evidence:
        visualize_reasoning_chain(G, patient_id, evidence)
        print(f"\n[Visualization] Reasoning chain saved to: reasoning_chain_{patient_id}.png")
    
    return llm_response, evidence

def main():
    print("="*80)
    print("KG-RAG POC for Obesity Management - UNIFIED Project")
    print("="*80)
    
    # 1. Build Graph
    G = kg_model.build_graph()
    print(f"\nKnowledge Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    # 2. Get all patient scenarios
    scenarios = kg_model.get_patient_scenarios()
    
    # 3. Process each patient
    for patient_id, info in scenarios.items():
        process_patient_query(G, patient_id, info['query'])
    
    # 4. Generate full graph visualization
    print(f"\n{'='*80}")
    print("Generating full Knowledge Graph visualization...")
    from visualize import visualize_graph
    visualize_graph(G, output_file="full_knowledge_graph.png")
    print("Full graph saved to: full_knowledge_graph.png")

if __name__ == "__main__":
    main()
