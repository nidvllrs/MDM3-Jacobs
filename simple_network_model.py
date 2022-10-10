#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 14:43:00 2022

@author: vickysmith
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt



# This is an Example of the imported data. Ideally, data will be imported in the same format, but can also be reformatted to look like this.
# A main assumption for the characteristics of the data is that the edgelist is sorted using the same order as the entries_exits dictionary (e.g [(A,..), (B,..), (C,..), etc])

edgelist= [('A', 'B'),('A', 'H'), ('A', 'D'),('B','C'),('B', 'H'), ('B', 'G'), ('B', 'F'), ('C', 'G'),('C','H'), ('D', 'B'), ('E', 'C'), ('E', 'F'),('G', 'F'), ('G', 'D')]
entries_exits ={'A':50,'B':60,'C':70,'D':20, 'E':100, 'F':40, 'G': 35, 'H': 100 }


def draw_graph(edgelist, entries_exits, origin, destination):
    """
    This function draws the graph of the original network, with the disrupted path highlighted in red.
    
    Input Parameters:
        - edgelist: list of edges between nodes
        - entries_exits: dictionary with a set of nodes and corresponding values for entries/exits (per day for simplicity)
        - origin, destination: the disrupted stops by the flood
        
    Output:
        - Original weighted and directed graph, showing the disrupted path in red
    """
    
    global nodes
    global G
    global weighted_edges
    
    # Define the list of nodes from the entries_exits dictionary:
    nodes= list(entries_exits)
    
    # Create the directed graph, adding the edges and nodes
    G = nx.DiGraph()
    G.add_edges_from(edgelist)
    G.add_nodes_from(nodes)
    
    
    # Define the disrupted path (includes two-way route):
    disrupted_path =[(origin, destination)]
    if origin in G.neighbors(destination):
        disrupted_path.append((destination, origin))
    
    # This allocates the entries/exits into the different ail lines:
    weights=[]
    for i in range(len(nodes)):                                                 # iterating through each node
        k = G.out_degree[nodes[i]]                                          
        while k != 0:                                                           
          weight = list(entries_exits.values())[i]//G.out_degree[nodes[i]]      # dividing entries/exits between number of leaving edges   
          weights.append(weight)                                                # adding these weights to an array k times (k = out degree of node i)
          k-=1                                                                  
          
    # This creates a dictionary allocating the weights to the correspondng edges     
    weighted_edges = {edgelist[k]:weights[k] for k in range(len(edgelist))}
    
    
    # Reassigns the edges to the graph, now with their corresponding weights.
    for i in range(len(edgelist)):
        G.add_edge(list(weighted_edges)[i][0], list(weighted_edges)[i][1], weight = list(weighted_edges.values())[i])
    
    # Draw graph:
    plt.figure(figsize=(10,10))
    pos = nx.circular_layout(G)

    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_nodes(G, pos, node_size = 400)
    nx.draw_networkx_labels(G, pos, font_size= 10)
    nx.draw_networkx_edges(G, pos, edgelist=edgelist,  edge_color='k', arrows=True, arrowsize=20, arrowstyle='->', width=1.0)
    nx.draw_networkx_edges(G, pos, edgelist=disrupted_path,  edge_color='r', arrows=True, arrowsize=20, arrowstyle='->', width=1.5)
    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels, font_size=10, font_color='k')

    plt.show()
    
    # This makes an adjacency matrix from the graph data:
    adj = nx.to_numpy_array(G)
    
    return nodes, weighted_edges, G, adj

  

def disruptions (G, weighted_edges, origin, destination):
    """
    This function calculates the number of people affected by the disruption, 
    calculates an alternative route these people would have to take, and displays it
    in a graph (aternative route in green)
    
    Input Parameters:
        - G: graph
        - weighted_edges: dictionary of edges and corresponding weights
        - origin, destination: the disrupted stops by the flood
        
    Output:
        - Weighted and directed graph, excluding the disrupted path,
        with the alternative route to this, in green 
        - Number of people affected by the disruption.
    """
    
    # This calculates the number of people affected, only considering people on the path between stops (inc. two-way)
    p_affected = weighted_edges[(origin,destination)]
    G.remove_edge(origin, destination)
    if origin in G.neighbors(destination):
        p_affected += weighted_edges[(destination,origin)]
        G.remove_edge(destination, origin)
    print( 'Number of people affected:\n', p_affected)
    
    
    # Calculate an alternative path
    alternative_path = nx.shortest_path(G, source = origin, target = destination)
    alternative_edges=[(alternative_path[i],alternative_path[i+1]) for i in range(len(alternative_path)-1) ]
    # Adding the number of reallocated people to the weight of the alternative path
    for (i,j) in alternative_edges:
        weighted_edges[(i,j)]+= p_affected
        G[i][j]['weight'] += p_affected
    
    
    # Draw graph
    plt.figure(figsize=(10,10))
    pos = nx.circular_layout(G)
   
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_nodes(G, pos, node_size = 400)
    nx.draw_networkx_labels(G, pos, font_size= 10)
    nx.draw_networkx_edges(G, pos,  edge_color='k', arrows=True, arrowsize=20, arrowstyle='->', width=1.0)
    nx.draw_networkx_edges(G, pos,edgelist=alternative_edges,  edge_color='g', arrows=True, arrowsize=20, arrowstyle='->', width=1.6)
    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels, font_size=10, font_color='k')
    
    plt.show()
    
    return p_affected

# Example of calling the functions:
    
draw_graph(edgelist, entries_exits, 'A', 'D') 

disruptions(G,weighted_edges, 'A', 'D')





    
