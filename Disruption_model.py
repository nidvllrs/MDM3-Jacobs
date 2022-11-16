#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 14:43:00 2022

@author: vickysmith
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import csv
tqdm.pandas()


def read_ids_network(csv):
    """
    This function converts the data of the station ids, and returns
    a dictionary of the stations and their ids, including the split nodes
    
    Input Parameters:
        - csv: filtered stations and station ids
        
    Output:
        - network_set: set of stations ids and station names including 'Splits'
    """
    i=1
    data = pd.read_csv(csv, header=1, low_memory=False)
    data = np.array(data)
    
    network_set = {}

    # Iterates through the data, extracting the relevant nodes
    for line in data:
        line = np.array(str(line).split(';'))
        line[-1]=''.join(e for e in line[-1] if e.isalnum())
        if line[-3]:
            network_set[line[-1]]=str(line[-3])
        # Assigns 'Split' to the nodes that don't have a 'station name' assigned to them
        else:
            network_set[line[-1]] = str('Split '+str(i))
            i+=1
        
    return network_set



  

def read_csv_passangers(csv, network_set):
    """
    This function converts the data in the edge flow csv into a 
    dictionary of edges and flow of passangers
    
    Input Parameters:
        - csv: edge_flows csv
        - network_set: set of stations ids and station names
        
    Output:
        - passengers: Dictionary of edges and the flow of passengers in the form: {edge: number of passengers}
    """
    # Read csv and convert it to array
    data = pd.read_csv(csv, header=2, low_memory=False)
    data = np.array(data)
    
    passengers = {}
    # Iterates through the data, extracting the relevant edges and their correspoding flow of passengers
    for line in data:
        line = np.array(str(line).split(';'))
        line[5] = line[5].replace('_','')
        line[6] = line[6].replace('_','')
        
        # Filter the data for the chosen network:
        if line[5] in list(network_set) and line[6] in list(network_set):
            origin = network_set[line[5]]
            destination = network_set[line[6]]
            n_passengers = int(float(line[7]))
            if n_passengers != 0:
                edge = (origin, destination)
                passengers[edge] = n_passengers
    edgelist = list(passengers)        
    return passengers, edgelist 


def draw_graph(edgelist, passengers, origin, destination):
    """
    This function draws the graph of the original network, with the disrupted path highlighted in red.
    
    Input Parameters:
        - edgelist: list of edges between nodes
        - passengers: dictionary of edges and corresponding of flow of passengers (weight)
        - origin, destination: the nodes of the disrupted edge by the flood 
        
    Output:
        - G: Original weighted and directed graph,
        - nodes: list of nodes in the Graph
        - adj: Adjacency matrix of G
        - Display of weighted and directed graph G, showing the disrupted edge in red
    """
    
    # Define the list of nodes from the list of edges:
    nodes = []
    for (i,j) in edgelist:
        if i not in nodes:
            nodes.append(i)
        if j not in nodes:
            nodes.append(j)
        
    # Separate stations and splits
    splits = [i for i in nodes if 'Split' in i]
    stations = [i for i in nodes if i not in splits]
    
    # Create the directed graph, and add the nodes
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    
                                                                                              
    # Define the disrupted path (two-way route):
    disrupted_path =[(origin, destination)]
    if origin in G.neighbors(destination):
        disrupted_path.append((destination, origin))

    # Flow of passangers from a to b = flow from b to a
    for (i,j) in edgelist:
        passengers[(j,i)] = passengers[(i,j)]   
    
   
    # Add edges and corresponding weights (number of passangers)        
    for (i,j) in edgelist:
        G.add_edge(i,j, weight = passengers[(i,j)])
    
    
    # Draw graph:
    fig, ax = plt.subplots(1, figsize=(90,90))
    pos = nx.spring_layout(G)
    
    nx.draw_networkx_nodes(G, pos=pos,ax=ax,  nodelist = stations, node_color="b", node_size = 3000, alpha=0.2, label='Stations')
    nx.draw_networkx_nodes(G, pos=pos,ax=ax, nodelist = splits, node_color="r", node_size = 1500, alpha=0.2, label='Splits')
    nx.draw_networkx_labels(G, pos=pos,ax=ax, font_size= 60)
    nx.draw_networkx_edges(G, pos=pos,ax=ax, edgelist = edgelist, arrowsize=40, arrowstyle='->' )
    # Draw disrupted path in red:
    nx.draw_networkx_edges(G, pos, edgelist= disrupted_path, arrowsize=40, width=8, arrowstyle='->', edge_color='r')
    
    edge_weights = nx.get_edge_attributes(G,'weight')
    edge_labels = {edge: edge_weights[edge] for edge in edgelist}
    nx.draw_networkx_edge_labels(G, pos=pos, ax=ax, edge_labels=edge_labels, rotate=False,  font_size= 30)
    
    plt.show()
    
    # This makes an adjacency matrix from the graph data:
    adj = nx.to_numpy_array(G)
    
    return nodes, G, adj 

  

def disruptions (G, nodes, passengers, origin, destination):
    """
    This function calculates the number of people affected  directly and indirectly by the disruption, 
    finds the shortest alternative route to take, and displays it
    in a graph (aternative route in green)
    
    Input Parameters:
        - G: Graph of original nerwork
        - nodes: nodes in the Graph G
        - passengers: dictionary of edges and corresponding of flow of passengers (weight)
        - origin, destination: the nodes of the disrupted edge by the flood 
        
    Output:
        - alternative_path: Shortest alternative path to the disrupted line
        - p_affected: Number of people affected directly by this disruption
        - ind_p_affected: Number of people affected indirectly by this disruption
        - Display of weighted and directed graph, excluding the disrupted path,
        with the alternative route in green 
    """
    # Separate stations and splits
    splits = [i for i in nodes if 'Split' in i]
    stations = [i for i in nodes if i not in splits]
    edgelist= [(i,j) for (i,j) in list(passengers) if i in nodes and j in nodes]
   
    # Remove path between origin and destination (+ return) from edge list and graph 
    edgelist.remove((origin, destination))
    if ((destination, origin)) in edgelist:
        edgelist.remove((destination, origin))
    
    G.remove_edge(origin, destination)
    if origin in G.neighbors(destination):
        G.remove_edge(destination, origin)  
        
    
    # Calculate the number of people directly affected
    p_affected = passengers[(origin,destination)]
    
    
    # Create an identical Graph without weights to find shortest path
    G2 = nx.DiGraph(seed=20)
    G2.add_nodes_from(nodes)
    for i in range(len(edgelist)):
        G2.add_edge(edgelist[i][0], edgelist[i][1], weight = 1)
    
    # Calculate an alternative path/edges in alternative path
    if nx.has_path(G2, source = origin, target = destination)== True:
        alternative_path = nx.shortest_path(G2, source = origin, target = destination)
        alternative_edges=[(alternative_path[i],alternative_path[i+1]) for i in range(len(alternative_path)-1)]
        # Alternative path only to include stations and not splits:
        alternative_path = [i for i in alternative_path if i in stations]
    # When edges might not have an alternative route:
    else :
       alternative_path = 'No alternative path'
       ind_p_affected = 0
       return alternative_path,  p_affected, ind_p_affected
       
    
    # Calculate number of people indirectly affected (people on alternative route)
    ind_p_affected = sum(passengers[i] for i in alternative_edges)
    
    # Adding the disrupted passangers to the alternative route
    for (i,j) in alternative_edges:
        passengers[(i,j)] +=  p_affected
    
    # Resetting graph with new edge weights / new nodes 
    G.remove_nodes_from([i for i in list(G.nodes) if i not in nodes])
    G.remove_edges_from( list(G.edges))
    for (i,j) in edgelist:
        G.add_edge(i,j, weight = passengers[(i,j)])  
    
    # Draw Graph
    plt.figure(figsize=(100,100))
    pos = nx.spring_layout(G)
    
    nx.draw_networkx_nodes(G, pos, nodelist = stations, node_color="b", node_size = 4000, alpha=0.2)
    nx.draw_networkx_nodes(G, pos, nodelist = splits, node_color="r", node_size = 1500, alpha=0.2)
    nx.draw_networkx_labels(G, pos, font_size= 40)
    
    alt_straight_edges = list(set(alternative_edges))
    edges = list(set(G.edges()) - set(alt_straight_edges))
    nx.draw_networkx_edges(G, pos, edgelist=edges, arrowsize=40, arrowstyle='->' )
    nx.draw_networkx_edges(G, pos, edgelist=alt_straight_edges, arrowsize=40, arrowstyle='->' , width=15, edge_color='g')
    
    edge_weights = nx.get_edge_attributes(G,'weight')
    edge_labels = {edge: edge_weights[edge] for edge in edgelist}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,rotate=False, font_size= 30)
    plt.show()

    return alternative_path,  p_affected, ind_p_affected


# CREATING NETWORK SET OF STATIONS & SPLITS
network_set = read_ids_network('manchester_ids.csv')    
# print('network_sett:\n',network_sett)


# CREATING EDGE LIST & DICTIONARY OF EDGES WITH FLOW OF PASSANGERS
passengers, edgelist = read_csv_passangers('edge_flows_1.csv', network_set)
# print('passengers:\n', passengers)


# DRAWING ORIGINAL NETWORK (DISRUPTED LINE IN RED)
nodes, G, adj = draw_graph(edgelist, passengers, 'Manchester Victoria', 'Salford Central')


# MODELLING DISRUPTION + ALTERNATIVE ROUTE GRAPH
alternative_path, p_affected, ind_p_affected = disruptions(G, nodes, passengers, 'Manchester Victoria', 'Salford Central')
print('alternative path:\n', alternative_path)
print('people affected directly:\n', p_affected)
print('people affected indirectly:\n', ind_p_affected)




# =============================================================================
#  FINDING THE EDGES WITH MORE PEOPLE AFFECTED IF DISRUPTED:
# =============================================================================

# ind_p_affected_set ={}
# p_affected_set ={}
# it = 0
# for (i,j) in edgelist:
#     network_sett = read_ids_network('manchester_ids.csv')    
#     passangers, edgelist = read_csv_passangers('edge_flows_1.csv', network_sett)
#     nodes, G, adj = draw_graph(edgelist, passangers, i, j )
#     alternative_path, p_affected, ind_p_affected = disruptions(G, nodes, passangers, i,j )
#     it += 1
#     p_affected_set[(i,j)]= p_affected
#     if (i,j) in ind_p_affected_set:
#         ind_p_affected_set[(i,j)] += ind_p_affected
#     else:
#         ind_p_affected_set[(i,j)] = ind_p_affected
#     print(it)
    
# p_affected_set=dict(sorted(p_affected_set.items(), key=lambda item: item[1],reverse=True))
# ind_p_affected_set=dict(sorted(ind_p_affected_set.items(), key=lambda item: item[1],reverse=True))
# print('p_affected_set: \n',p_affected_set)
# print('ind_p_affected_set: \n',ind_p_affected_set)


# Saving data of edges + disruptions caused in csv

# with open('p_affected_data.csv', mode='w') as file:
#     writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     writer.writerow(['Origin', 'Destination', 'People affected directly'])
#     for (i,j) in list(p_affected_set):
#         writer.writerow([i,j, p_affected_set[(i,j)]])

# with open('ind_p_affected_data.csv', mode='w') as file:
#     writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     writer.writerow(['Origin', 'Destination', 'People affected indirectly'])
#     for (i,j) in list(ind_p_affected_set):
#         writer.writerow([i,j, ind_p_affected_set[(i,j)]])

