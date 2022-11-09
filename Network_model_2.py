#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 14:43:00 2022

@author: vickysmith
"""

import networkx as nx
#import mynetworkx.py as my_nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import re
import ast
import configparser
from tqdm import tqdm
tqdm.pandas()


# This is an Example of the imported data. Ideally, data will be imported in the same format, but can also be reformatted to look like this.
# A main assumption for the characteristics of the data is that the edgelist is sorted using the same order as the entries_exits dictionary (e.g [(A,..), (B,..), (C,..), etc])

# edgelist= [('A', 'B'),('A', 'H'), ('A', 'D'),('B','C'),('B', 'H'), ('B', 'G'), ('B', 'F'), ('C', 'G'),('C','H'), ('D', 'B'), ('E', 'C'), ('E', 'F'),('G', 'F'), ('G', 'D')]
# edgelist= [('Bedminster', 'Bristol Temple Meads'),('Bedminster', 'Sea Mills'), ('Bedminster', 'Lawrence Hill'),('Bristol Temple Meads','Clifton Down'),
# ('Bristol Temple Meads', 'Sea Mills'), ('Bristol Temple Meads', 'Redland'), ('Bristol Temple Meads', 'Parson Street'), ('Clifton Down', 'Redland'),('Clifton Down','Sea Mills'),
#   ('Lawrence Hill', 'Bristol Temple Meads'), ('Montpelier', 'Clifton Down'), ('Montpelier', 'Parson Street'),('Redland', 'Parson Street'), ('Redland', 'Lawrence Hill'), 
#   ('Sea Mills', 'Redland'), ('Shirehampton', 'Clifton Down'), ('Shirehampton', "St.Andrew's Road"), ("St.Andrew's Road", 'Stapleton Road')]

#entries_exits ={'A':50,'B':60,'C':70,'D':20, 'E':100, 'F':40, 'G': 35, 'H': 100 }

network_set = {'railn3208' : 'Bedminster'   ,   'railn53' : 'Bristol Temple Meads',
               'railn2454' : 'Clifton Down' , 'railn1889' : 'Lawrence Hill',
               'railn2453' : 'Montpelier'   , 'railn3209' : 'Parson Street' ,
               'railn2452' : 'Redland'      , 'railn3353' : 'Stapleton Road',
               'railn3288' : 'Sea Mills'    , 'railn3121' : 'Shirehampton',
               'railn210'  : 'Split 1'      , 'railn54'   : 'Split 2',   
               'railn55'   : 'Split 3'      , 'railn3029' : 'Split 4',
               'railn208'  : 'Split 5'      ,  'railn209' : 'Split 6',
               'railn3916' : 'Split 7'}
 
def my_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items


def read_ids_network(csv):
    
    i=1
    data = pd.read_csv(csv, header=1, low_memory=False)
    data = np.array(data)
    
    network_sett = {}
    # Iterates through the data, extracting the relevant edges and their correspoding flow of passangers
    for line in data:
        line = np.array(str(line).split(';'))
        line[-1]=''.join(e for e in line[-1] if e.isalnum())
        
        if line[-3]:
            network_sett[line[-1]]=str(line[-3])
        else:
            network_sett[line[-1]] = str('Split '+str(i))
            i+=1
        
    return network_sett



def read_matrix (matrixod, network_set):
    """
    This function reads the data in the filtered od matrix and creates a list of edges between stations
    and a dictionary of these edges and the distance between them.
    
    Input Parameters:
        - csv: filtered od matrix csv
        - network_set: set of stations ids and station names
        
    Output:
        - edgelist: set of edges
        - distance_edges: Dictionary of edges and the distnace between them in the form: {edge: distance}
    """
    
    # Reads csv and converts it to numpy array
    data = pd.read_csv(matrixod, header=1, on_bad_lines='skip', low_memory=False)
    print(data.head())
    data = np.array(data) 
    
    # Creates an array of edges between stations(using their node_ids)
    edge_ids =[] 
    for line in data:
        print(line[4])
        line[4]= np.array(line[4].split(','))
        line[4] = [''.join(e for e in string if e.isalnum()) for string in line[3]]
        for i in range(len(line[4])-1):
            if (line[4][i], line[4][i+1]) not in edge_ids:
                print((line[4][i], line[4][i+1]))
                edge_ids.append((line[4][i], line[4][i+1]))
     

    # Removes paths between stations that are not considered
    for (i,j) in edge_ids:
        if i not in network_set.keys() or j not in network_set.keys():
            edge_ids.remove((i,j))
            
    # Converts the edge list from node_ids to station names  
    edgelist=[]       
    for (i,j) in edge_ids:
        if i in network_set.keys() and j in network_set.keys():
            (u,v) = (network_set[i], network_set[j])
            edgelist.append((u,v))
   
    # Creates dictionary of paths and corresponding distances between them
    distance_edges ={}
    for line in data:        
        if len(line[3])==2:
              if line[3][0] in network_set.keys() and line[3][1] in network_set.keys():
                  distance_edges[(network_set[line[3][0]], network_set[line[3][1]])]= line[5]           
    
    return edgelist, distance_edges



  

def read_csv_passangers(csv, network_set, edgelist):
    """
    This function converts the data in the edge flow csv into a 
    dictionary of edges and flow of passangers, for the Bristol network.
    
    Input Parameters:
        - csv: edge flow csv
        - network_set: set of stations ids and station names
        - edgelist: set of edges 
        
    Output:
        - passangers: Dictionary of edges and the number of passangers in the form: {edge: number of passangers}
    """
    # Read csv and convert it to array
    data = pd.read_csv(csv, header=2, low_memory=False)
    data = np.array(data)
    
    passangers = {}
    # Iterates through the data, extracting the relevant edges and their correspoding flow of passangers
    for line in data:
        line = np.array(str(line).split(';'))
        line[5] = line[5].replace('_','')
        line[6] = line[6].replace('_','')
        
        # Filter the data for the Bristol network:
        if line[5] in list(network_set) and line[6] in list(network_set):
            origin = network_set[line[5]]
            destination = network_set[line[6]]
            n_passangers = int(float(line[7]))
            if n_passangers != 0:
                edge = (origin, destination)
                passangers[edge] = n_passangers
            
    return passangers    


def draw_graph(edgelist, passangers):#, origin, destination):
    """
    This function draws the graph of the original network, # with the disrupted path highlighted in red.
    
    Input Parameters:
        - edgelist: list of edges between nodes
        - passangers: dictionary of edges and corresponding of flow of passangers 
        - # origin, destination: the disrupted stops by the flood 
        
    Output:
        - Original weighted and directed graph, # showing the disrupted path in red
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
    edgelist = list(passangers)
    
    # Create the directed graph, adding the edges and nodes
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    
                                                                                              
    # Define the disrupted path (includes two-way route):
    # disrupted_path =[(origin, destination)]
    # if origin in G.neighbors(destination):
    #     disrupted_path.append((destination, origin))
     
    # Flow of passangers from a to b = flow from b to a
    for (i,j) in edgelist:
        passangers[(j,i)] = passangers[(i,j)]
    
    
    # Add edges and corresponding weights (number of passangers)        
    for i in range(len(edgelist)):
        G.add_edge(edgelist[i][0], edgelist[i][1], weight = list(passangers.values())[i])
    
    
    # Draw graph:
    fig, ax = plt.subplots(1, figsize=(80,80))
    pos = nx.planar_layout(G, scale=0.5)
    
    nx.draw_networkx_nodes(G, pos=pos,ax=ax,  nodelist = stations, node_color="b", node_size = 6000, alpha=0.5, label='Stations')
    nx.draw_networkx_nodes(G, pos=pos,ax=ax, nodelist = splits, node_color="r", node_size = 3000, alpha=0.5, label='Splits')
    nx.draw_networkx_labels(G, pos=pos,ax=ax, font_size= 60)
    nx.draw_networkx_edges(G, pos=pos,ax=ax, edgelist=edgelist, arrowsize=40, arrowstyle='->' )
    # Draw disrupted path in red:
    #nx.draw_networkx_edges(G, pos, edgelist= disrupted_path, arrowsize=40, width=12, arrowstyle='->', edge_color='r')
    
    edge_weights = nx.get_edge_attributes(G,'weight')
    edge_labels = {edge: edge_weights[edge] for edge in edgelist}
    nx.draw_networkx_edge_labels(G, pos=pos, ax=ax, edge_labels=edge_labels, rotate=False,  font_size= 50)
    
    #ax.legend(scatterpoints = 1, loc='lower right',)
    plt.show()
    
    # This makes an adjacency matrix from the graph data:
    adj = nx.to_numpy_array(G)
    
    return nodes, G, adj 

  

def disruptions (G, nodes, passangers, origin, destination, distance_edges):
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
    # Separate stations and splits
    splits = [i for i in nodes if 'Split' in i]
    stations = [i for i in nodes if i not in splits]
    edgelist= list(passangers)
    
    # This calculates the number of people affected, only considering people on the path between stops (inc. two-way)
    p_affected = passangers[(origin,destination)]
    G.remove_edge(origin, destination)
    if origin in G.neighbors(destination):
        p_affected += passangers[(destination,origin)]
        G.remove_edge(destination, origin)
   
    
    # Create an identical Graph, weighted instead by the distance between nodes (to find shortest path)
    G2 = nx.DiGraph()
    G2.add_nodes_from(nodes)
    for i in range(len(edgelist)):
        G2.add_edge(edgelist[i][0], edgelist[i][1], weight = list(distance_edges.values())[i])
    
        
    # Calculate an alternative path
    alternative_path = nx.shortest_path(G2, source = origin, target = destination)
    alternative_edges=[(alternative_path[i],alternative_path[i+1]) for i in range(len(alternative_path)-1)]
   
   
    
    # Adding the number of reallocated people to the weight of the alternative path
    for (i,j) in alternative_edges:
        passangers[(i,j)]+= p_affected
        G[i][j]['weight'] += p_affected
        
    # Draw Graph
    plt.figure(figsize=(100,100))
    pos = nx.spring_layout(G, scale=0.5)
    
    nx.draw_networkx_nodes(G, pos, nodelist = stations, node_color="b", node_size = 6000, alpha=0.4)
    nx.draw_networkx_nodes(G, pos, nodelist = splits, node_color="r", node_size = 6000, alpha=0.4)
    nx.draw_networkx_labels(G, pos, font_size= 50)
    # curved_edges = [(i,j) for (i,j) in G.edges() if 'Split' in i and 'Split' in j ]
    # alt_curved_edges = [(i,j) for (i,j) in alternative_edges if 'Split' in i and 'Split' in j ]
    # curved_edges = list(set(curved_edges) - set(alt_curved_edges))
    
    alt_straight_edges = list(set(alternative_edges))
    straight_edges = list(set(G.edges()) - set(alt_straight_edges))
    nx.draw_networkx_edges(G, pos, edgelist=straight_edges, arrowsize=40, arrowstyle='->' )
    nx.draw_networkx_edges(G, pos, edgelist=alt_straight_edges, arrowsize=40, arrowstyle='->' , width=15, edge_color='g')
    # arc_rad = 0.35
    # nx.draw_networkx_edges(G, pos, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}', arrowsize=40, arrowstyle='->' )
    # nx.draw_networkx_edges(G, pos, edgelist=alt_curved_edges, connectionstyle=f'arc3, rad = {arc_rad}', arrowsize=40, width=15, arrowstyle='->', edge_color='g' )
   
    edge_weights = nx.get_edge_attributes(G,'weight')
    # curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
    edge_labels = {edge: edge_weights[edge] for edge in edgelist}
    # my_draw_networkx_edge_labels(G, pos, edge_labels=curved_edge_labels,rotate=False,rad = arc_rad, font_size= 30)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,rotate=False, font_size= 30)
    plt.show()
    

    return p_affected

# Running the code:
# network_sett = read_ids_network('manchester_ids.csv')    
# print(network_sett)
# edgelist, distance_edges = read_matrix('manchester_od_matrix.csv', network_sett)

# passangers= read_csv_passangers('edge_flows_1.csv', network_set, edgelist)

# nodes, G, adj = draw_graph(edgelist, passangers)# 'Bristol Temple Meads', 'Split 2') 

#disruptions(G, nodes, passangers, distance_edges, 'Bristol Temple Meads', 'Split 2')

data = pd.read_csv('manchester_od_matrix.csv', header=None, on_bad_lines='skip', low_memory=False)

data = np.array(data) 
#data = data.values
print(data)


    
