import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualize_graph(G, highlighted_paths=None, output_file="knowledge_graph.png"):
    """
    Visualizes the Knowledge Graph with optional path highlighting.
    
    Args:
        G: NetworkX DiGraph
        highlighted_paths: List of tuples representing edges to highlight [(source, target), ...]
        output_file: Output filename for the visualization
    """
    plt.figure(figsize=(20, 14))
    
    # Create layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Color nodes by type
    node_colors = []
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'unknown')
        if node_type == 'Patient':
            node_colors.append('#FF6B6B')  # Red
        elif node_type == 'Preference':
            node_colors.append('#4ECDC4')  # Teal
        elif node_type == 'Behavior':
            node_colors.append('#FFE66D')  # Yellow
        elif node_type == 'Outcome':
            node_colors.append('#95E1D3')  # Light green
        elif node_type == 'Metric':
            node_colors.append('#C7CEEA')  # Light purple
        else:
            node_colors.append('#CCCCCC')  # Gray
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, alpha=0.9)
    
    # Draw regular edges
    regular_edges = [(u, v) for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, edgelist=regular_edges, 
                          edge_color='gray', arrows=True, 
                          arrowsize=20, width=1.5, alpha=0.5)
    
    # Highlight specific paths if provided
    if highlighted_paths:
        nx.draw_networkx_edges(G, pos, edgelist=highlighted_paths,
                              edge_color='red', arrows=True,
                              arrowsize=25, width=4, alpha=0.9)
    
    # Draw labels
    labels = {node: node.replace('_', '\n') for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
    
    # Draw edge labels for important relationships
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        relation = data.get('relation', '')
        if relation in ['conflicts_with', 'causes', 'influences']:
            edge_labels[(u, v)] = relation
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
    
    # Create legend
    legend_elements = [
        mpatches.Patch(color='#FF6B6B', label='Patient'),
        mpatches.Patch(color='#4ECDC4', label='Preference (PPI)'),
        mpatches.Patch(color='#FFE66D', label='Behavior (DHT)'),
        mpatches.Patch(color='#95E1D3', label='Outcome (COA)'),
        mpatches.Patch(color='#C7CEEA', label='Metric (COA)')
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.title("Knowledge Graph: Obesity Management (UNIFIED Project)", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Graph visualization saved to: {output_file}")
    plt.close()

def visualize_reasoning_chain(G, patient_id, evidence_chains, output_file=None):
    """
    Visualizes the knowledge graph with highlighted reasoning chains for a specific patient.
    
    Args:
        G: NetworkX DiGraph
        patient_id: Patient node ID
        evidence_chains: List of evidence dictionaries from RAG retrieval
        output_file: Optional output filename
    """
    if output_file is None:
        output_file = f"reasoning_chain_{patient_id}.png"
    
    # Extract edges to highlight from evidence chains
    highlighted_edges = []
    for chain in evidence_chains:
        highlighted_edges.append((chain['preference'], chain['behavior']))
        highlighted_edges.append((chain['behavior'], chain['outcome']))
    
    visualize_graph(G, highlighted_paths=highlighted_edges, output_file=output_file)

if __name__ == "__main__":
    from kg_model import build_graph
    
    G = build_graph()
    visualize_graph(G, output_file="full_knowledge_graph.png")
    print("Full knowledge graph visualization created.")
