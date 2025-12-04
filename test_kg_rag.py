import pytest
import networkx as nx
from kg_model import build_graph, get_patient_scenarios
from rag_engine import retrieve_context, format_prompt

class TestKnowledgeGraph:
    """Test suite for Knowledge Graph construction."""
    
    def test_graph_construction(self):
        """Test that graph is built correctly."""
        G = build_graph()
        assert G.number_of_nodes() > 0, "Graph should have nodes"
        assert G.number_of_edges() > 0, "Graph should have edges"
        
    def test_patient_nodes_exist(self):
        """Test that all patient nodes exist."""
        G = build_graph()
        patients = ["Patient_A", "Patient_B", "Patient_C"]
        for patient in patients:
            assert patient in G.nodes(), f"{patient} should exist in graph"
            assert G.nodes[patient]['type'] == 'Patient', f"{patient} should be of type Patient"
    
    def test_conflict_relationships(self):
        """Test that conflict relationships are properly defined."""
        G = build_graph()
        conflicts = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') == 'conflicts_with']
        assert len(conflicts) > 0, "Graph should have conflict relationships"
        
        # Check that conflicts have reasons
        for u, v in conflicts:
            edge_data = G.get_edge_data(u, v)
            assert 'reason' in edge_data, f"Conflict edge {u}->{v} should have a reason"

class TestRAGEngine:
    """Test suite for RAG retrieval logic."""
    
    def test_retrieve_context_patient_a(self):
        """Test context retrieval for Patient A."""
        G = build_graph()
        reasoning_text, evidence = retrieve_context(G, "Patient_A")
        
        assert len(reasoning_text) > 0, "Should retrieve reasoning text"
        assert len(evidence) > 0, "Should retrieve evidence chains"
        
        # Check that evidence contains expected structure
        for chain in evidence:
            assert 'preference' in chain
            assert 'behavior' in chain
            assert 'outcome' in chain
    
    def test_retrieve_context_patient_b(self):
        """Test context retrieval for Patient B."""
        G = build_graph()
        reasoning_text, evidence = retrieve_context(G, "Patient_B")
        
        assert len(reasoning_text) > 0, "Should retrieve reasoning text"
        assert len(evidence) > 0, "Should retrieve evidence chains"
    
    def test_retrieve_context_patient_c(self):
        """Test context retrieval for Patient C."""
        G = build_graph()
        reasoning_text, evidence = retrieve_context(G, "Patient_C")
        
        assert len(reasoning_text) > 0, "Should retrieve reasoning text"
        assert len(evidence) > 0, "Should retrieve evidence chains"
    
    def test_retrieve_context_invalid_patient(self):
        """Test context retrieval for non-existent patient."""
        G = build_graph()
        reasoning_text, evidence = retrieve_context(G, "Patient_Z")
        
        assert "not found" in reasoning_text[0].lower(), "Should indicate patient not found"
        assert len(evidence) == 0, "Should return no evidence for invalid patient"
    
    def test_format_prompt(self):
        """Test prompt formatting."""
        G = build_graph()
        reasoning_text, _ = retrieve_context(G, "Patient_A")
        query = "Why is the patient experiencing slow weight loss?"
        
        prompt = format_prompt(query, reasoning_text)
        
        assert query in prompt, "Prompt should contain the query"
        assert "clinical" in prompt.lower(), "Prompt should reference clinical context"

class TestScenarios:
    """Test suite for patient scenarios."""
    
    def test_get_patient_scenarios(self):
        """Test that patient scenarios are properly defined."""
        scenarios = get_patient_scenarios()
        
        assert len(scenarios) == 3, "Should have 3 patient scenarios"
        assert "Patient_A" in scenarios
        assert "Patient_B" in scenarios
        assert "Patient_C" in scenarios
        
        # Check structure
        for patient_id, info in scenarios.items():
            assert 'query' in info, f"{patient_id} should have a query"
            assert 'description' in info, f"{patient_id} should have a description"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
