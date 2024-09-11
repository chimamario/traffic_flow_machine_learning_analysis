import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

adj_matrix = pd.read_csv(r"/Users/mariochima/Desktop/my first folder/coding folder/machine learning practice/traffic flow/adj_matrix.csv", index_col=0)

adj_matrix_np = adj_matrix.to_numpy()

G = nx.from_numpy_array(adj_matrix_np)

degree_centrality = nx.degree_centrality(G) # Degree centrality (returns a dictionary where keys are nodes and values are centrality scores)
clustering_coefficient = nx.clustering(G) # Clustering coefficient (measures the degree to which nodes cluster together)
eigenvector_centrality = nx.eigenvector_centrality(G) # Eigenvector centrality (measure of node influence)
shortest_paths = dict(nx.shortest_path_length(G)) # Shortest paths (returns the lengths of the shortest paths between nodes)
betweenness_centrality = nx.betweenness_centrality(G) # Betweenness centrality (measures the amount of control a node has over communication between pairs of other nodes)
connected_components = list(nx.connected_components(G)) # Connected components (returns a list of connected components)

component_sizes = {node: len(component) for component in connected_components for node in component}
connected_component_size_list = [component_sizes[node] for node in G.nodes()]

# Combine these features into a DataFrame or use them as inputs to a machine learning model
graph_features = pd.DataFrame({
    'degree_centrality': degree_centrality.values(), #36 values
    'clustering_coefficient': clustering_coefficient.values(), #36 values
    'eigenvector_centrality': eigenvector_centrality.values(), #36 values
    'betweenness_centrality': betweenness_centrality.values(), #36 values
    'connected_component_size': connected_component_size_list,
})

# print(graph_features)

G = nx.from_numpy_array(adj_matrix_np)

# Draw the graph
plt.figure(figsize=(12, 12))  # Set figure size
pos = nx.spring_layout(G)  # Spring layout for a visually appealing arrangement

# Draw nodes and edges
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7)
nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

# Display the graph
plt.title("Graph Visualization Based on Adjacency Matrix")
plt.show()




