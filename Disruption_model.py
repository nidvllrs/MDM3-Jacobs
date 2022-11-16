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
    
   
    # For zoomed in networks (for visualisation):
    # nodes =  ['Split 8', 'Split 9', 'Split 2',
    #             'Split 5',
    #             'Split 6', 'Manchester Oxford Road',
    #             'Deansgate', 'Manchester Piccadilly', 'Ashburys',
    #             'Split 10', 'Split 11',  'Split 13', 
    #             'Split 16', 'Split 17', 'Split 18', 
    #             'Gorton', 'Fairfield', 
    #             'Split 21', 'Split 22', 'Guide Bridge', 
    #             'Split 24', 'Ashton-under-Lyne',  
    #             'Stalybridge','Split 25', 'Split 33',
    #             'Salford Central', 'Split 29', 
    #             'Manchester Victoria', 'Eccles', 'Split 2', 'Ardwick', 'Split 12',
    #             'Levenshulme', 'Mauldeth Road', 'Heaton Chapel', 'Split 28', 'Split 30',
    #             'Flowery Field']

    # splits = [i for i in nodes if 'Split' in i]
    # stations = [i for i in nodes if i not in splits]
    # edgelist = [(i,j) for (i,j) in edgelist if i in nodes and j in nodes]
    
    
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
# CALCULATING CENTRALITY FOR EACH NODE:
# centrality = nx.closeness_centrality(G)
# centrality=dict(sorted(centrality.items(), key=lambda item: item[1],reverse=True))
# print('centrality\n', centrality)
# =============================================================================


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


# ========================================================================
# =============================================================================
# Top edge with most people directly affected : ('Split 25', 'Manchester Piccadilly') : 109496
#                       between stations only : ('Levenshulme', 'Heaton Chapel')  : 80856
# ^ This can be done by sorting passangers*

# Top edge affected indirectly  : ('Split 24', 'Split 29') : 727350
#         between stations only : ('Gorton', 'Fairfield')  : 724237
# ^ These are vulnerable routes because they lie in many alternative routes 
# =============================================================================





# =============================================================================
# DEGREE CENTRALITY:
# =============================================================================
# =============================================================================
# centrality
# {'Split 33': 0.10810810810810811,           'Split 8': 0.08108108108108109,
#  'Split 2': 0.08108108108108109,            'Split 3': 0.08108108108108109,
#  'Split 11': 0.08108108108108109,           'Split 12': 0.08108108108108109,
#  'Split 13': 0.08108108108108109,           'Split 24': 0.08108108108108109,
#  'Split 25': 0.08108108108108109,           'Salford Central': 0.08108108108108109,
#  'Split 35': 0.08108108108108109,           'Split 9': 0.05405405405405406, 
#  'Salford Crescent': 0.05405405405405406,   'Split 4': 0.05405405405405406,
#  'Split 5': 0.05405405405405406,            'Split 7': 0.05405405405405406,
#  'Ryder Brow': 0.05405405405405406,         'Belle Vue': 0.05405405405405406, 
#  'Split 6': 0.05405405405405406,            'Manchester Oxford Road': 0.05405405405405406, 
#  'Deansgate': 0.05405405405405406,          'Manchester Piccadilly': 0.05405405405405406,
#  'Ashburys': 0.05405405405405406,           'Split 10': 0.05405405405405406, 
#  'Levenshulme': 0.05405405405405406,        'Heaton Chapel': 0.05405405405405406,
#  'Navigation Road': 0.05405405405405406,    'Chassen Road': 0.05405405405405406, 
#  'Newton for Hyde': 0.05405405405405406,    'Romiley': 0.05405405405405406,
#  'Split 14': 0.05405405405405406,           'Split 15': 0.05405405405405406, 
#  'Split 16': 0.05405405405405406,           'Split 17': 0.05405405405405406, 
#  'Split 18': 0.05405405405405406,           'Stockport': 0.05405405405405406,
#  'Swinton (Greater Manchester)': 0.05405405405405406, 
#  'Moorside': 0.05405405405405406,           'Mauldeth Road': 0.05405405405405406,
#  'Burnage': 0.05405405405405406,            'Humphrey Park': 0.05405405405405406,
#  'Urmston': 0.05405405405405406,            'Trafford Park': 0.05405405405405406, 
#  'Gorton': 0.05405405405405406,             'Fairfield': 0.05405405405405406, 
#  'Brinnington': 0.05405405405405406,        'Reddish North': 0.05405405405405406,
#  'Bredbury': 0.05405405405405406,           'Split 21': 0.05405405405405406, 
#  'Split 22': 0.05405405405405406,           'Guide Bridge': 0.05405405405405406, 
#  'Ashton-under-Lyne': 0.05405405405405406,  'East Didsbury': 0.05405405405405406,
#  'Stalybridge': 0.05405405405405406,        'Eccles': 0.05405405405405406, 
#  'Ardwick': 0.05405405405405406,            'Split 28': 0.05405405405405406, 
#  'Split 30': 0.05405405405405406,           'Split 32': 0.05405405405405406,
#  'Split 34': 0.05405405405405406,           'Split 29': 0.05405405405405406, 
#  'Flowery Field': 0.05405405405405406,      'Manchester Victoria': 0.05405405405405406, 
#  'Davenport': 0.05405405405405406,          'Manchester United Football Ground': 0.05405405405405406,
#  'Altrincham': 0.02702702702702703,         'Flixton': 0.02702702702702703, 
#  'Godley': 0.02702702702702703,             'Split 1': 0.02702702702702703, 
#  'Clifton': 0.02702702702702703,            'Walkden': 0.02702702702702703, 
#  'Patricroft': 0.02702702702702703,         'Gatley': 0.02702702702702703,
#  'Moston': 0.02702702702702703,              'Woodsmoor': 0.02702702702702703}



# =============================================================================
# CLOSENESS CENTRALITY:
# =============================================================================
# =============================================================================
# centrality
 # {'Split 25': 0.13479052823315119,                  'Manchester Piccadilly': 0.13214285714285715,
 #  'Manchester Oxford Road': 0.1295971978984238,     'Split 10': 0.12937062937062938,
 #  'Deansgate': 0.12714776632302405,                 'Ardwick': 0.12542372881355932,
 #  'Split 13': 0.12478920741989882,                  'Ashburys': 0.12436974789915967,
 #  'Split 33': 0.12012987012987013,                  'Split 11': 0.11974110032362459, 
 #  'Split 12': 0.11690363349131122,                  'Salford Central': 0.115625, 
 #  'Split 2': 0.11297709923664122,                   'Manchester United Football Ground': 0.11280487804878049, 
 #  'Gorton': 0.1126331811263318,                     'Manchester Victoria': 0.11028315946348734,
 #  'Belle Vue': 0.10930576070901034,                 'Levenshulme': 0.10787172011661808,
 #  'Eccles': 0.10771470160116449,                    'Fairfield': 0.10755813953488372,
 #  'Mauldeth Road': 0.10571428571428572,             'Split 6': 0.10541310541310542,
 #  'Salford Crescent': 0.10306406685236769,          'Split 22': 0.10292072322670376, 
 #  'Trafford Park': 0.10263522884882108,             'Split 8': 0.1009549795361528, 
 #  'Ryder Brow': 0.1002710027100271,                 'Heaton Chapel': 0.09986504723346828, 
 #  'Guide Bridge': 0.09866666666666667,              'Patricroft': 0.09736842105263158, 
 #  'Burnage': 0.09622886866059818,                   'Split 9': 0.09585492227979274,
 #  'Split 24': 0.09560723514211886,                  'Split 3': 0.09450830140485313, 
 #  'Humphrey Park': 0.09390862944162437,             'Split 17': 0.0930817610062893, 
 #  'Split 29': 0.09284818067754078,                  'Split 15': 0.09273182957393483, 
 #  'Split 5': 0.0925,                                'Reddish North': 0.09238451935081149, 
 #  'Stalybridge': 0.09158415841584158,               'Split 16': 0.09046454767726161, 
 #  'Ashton-under-Lyne': 0.09035409035409035,         'Split 21': 0.0891566265060241, 
 #  'Split 28': 0.08820023837902265,                  'East Didsbury': 0.0880952380952381, 
 #  'Split 18': 0.08799048751486326,                  'Swinton (Greater Manchester)': 0.08685446009389672, 
 #  'Clifton': 0.08644859813084112,                   'Stockport': 0.08634772462077013, 
 #  'Urmston': 0.08634772462077013,                   'Brinnington': 0.08545034642032333, 
 #  'Split 4': 0.08515535097813579,                   'Split 30': 0.08167770419426049, 
 #  'Gatley': 0.08105147864184009,                    'Split 35': 0.08061002178649238, 
 #  'Moorside': 0.08017334777898158,                  'Chassen Road': 0.07974137931034483,
 #  'Bredbury': 0.07931404072883172,                  'Split 7': 0.07872340425531915, 
 #  'Flowery Field': 0.0758974358974359,              'Split 34': 0.0751269035532995,
 #  'Davenport': 0.07482305358948432,                 'Walkden': 0.07429718875502007, 
 #  'Flixton': 0.07392607392607392,                   'Split 14': 0.07385229540918163, 
 #  'Moston': 0.07305034550839092,                    'Newton for Hyde': 0.07074569789674952,
 #  'Split 32': 0.07020872865275142,                  'Woodsmoor': 0.0696798493408663, 
 #  'Romiley': 0.06896551724137931,                   'Godley': 0.06613047363717604, 
 #  'Navigation Road': 0.06577777777777778,           'Split 1': 0.06457242582897033,
 #  'Altrincham': 0.06176961602671119}






# =============================================================================
# BETWEENNESS CENTRALITY:
# =============================================================================
# =============================================================================
# centrality
# {'Split 25': 0.5388744909292855, 'Split 13': 0.41873380229544616,
 # 'Manchester Piccadilly':0.37430581266197704,'Manchester Oxford Road': 0.37023324694557574, 
 # 'Split 11': 0.36690114772306553,           'Deansgate': 0.3661606812291744, 
 # 'Split 12': 0.34394668641243986,           'Ardwick': 0.3435764531654943, 
 # 'Split 33': 0.33598667160310997,           'Split 10': 0.3320992225101814, 
 # 'Ashburys': 0.3235838578304332,            'Levenshulme': 0.23694927804516847,
 # 'Salford Central': 0.22991484635320253,    'Split 8': 0.22139948167345427,
 # 'Manchester Victoria': 0.2184376156978897, 'Heaton Chapel': 0.2165864494631618,
 # 'Split 24': 0.21121806738245094,           'Gorton': 0.20955201777119586, 
 # 'Split 6': 0.20696038504257683,            'Fairfield': 0.19733432062199185,
 # 'Split 15': 0.195483154387264,             'Split 22': 0.18585708996667902, 
 # 'Guide Bridge': 0.17437985931136618,       'Belle Vue': 0.173639392817475,
 # 'Stockport': 0.173639392817475,            'Split 35': 0.15401703072935952, 
 # 'Split 2': 0.1510551647537949,             'Ryder Brow': 0.1510551647537949, 
 # 'Split 9': 0.13069233617178824,            'Salford Crescent': 0.12773047019622363, 
 # 'Reddish North': 0.12773047019622363,      'Manchester United Football Ground': 0.12773047019622363,
 # 'Split 29': 0.12106627175120327,           'Split 17': 0.12069603850425768, 
 # 'Stalybridge': 0.1158830062939652,         'Split 16': 0.11218067382450944,
 # 'Ashton-under-Lyne': 0.11181044057756387, N'Split 21': 0.10773787486116254, 
 # 'Split 18': 0.1068122917437986,            'Split 3': 0.10477600888559793, 
 # 'Trafford Park': 0.1036653091447612,       'Brinnington': 0.1036653091447612,
 # 'Split 28': 0.1036653091447612,            'Split 5': 0.07885968159940764, 
 # 'Mauldeth Road': 0.07885968159940764,      'Humphrey Park': 0.07885968159940764,
 # 'Bredbury': 0.07885968159940764,           'Split 30': 0.07885968159940764, 
 # 'Split 34': 0.07885968159940764,           'Split 4': 0.053313587560162905, 
 # 'Split 14': 0.053313587560162905,          'Swinton (Greater Manchester)': 0.053313587560162905, 
 # 'Burnage': 0.053313587560162905,           'Urmston': 0.053313587560162905, 
 # 'Split 32': 0.053313587560162905,          'Flowery Field': 0.053313587560162905, 
 # 'Split 7': 0.02702702702702703,            'Navigation Road': 0.02702702702702703, 
 # 'Chassen Road': 0.02702702702702703,       'Newton for Hyde': 0.02702702702702703,
 # 'Romiley': 0.02702702702702703,            'Moorside': 0.02702702702702703, 
 # 'East Didsbury': 0.02702702702702703,      'Eccles': 0.02702702702702703,
 # 'Davenport': 0.02702702702702703,          'Altrincham': 0.0, 
 # 'Flixton': 0.0, 'Godley': 0.0, 'Split 1': 0.0, 
 # 'Clifton': 0.0, 'Walkden': 0.0, 'Patricroft': 0.0,
 # 'Gatley': 0.0, 'Moston': 0.0, 'Woodsmoor': 0.0}
# =============================================================================




# =============================================================================
# EIGENVECTOR CENTRALITY:
# =============================================================================
# =============================================================================
# centrality
 # {'Split 33': 0.5347615154913689,                       'Split 2': 0.44557718592922146,
 #  'Salford Central': 0.44359892895160974,               'Split 13': 0.3034143420830512,
 #  'Eccles': 0.2333664422049024,                         'Salford Crescent': 0.2097847261066507,
 #  'Manchester Victoria': 0.2025312478122092,            'Deansgate': 0.13732963933959488, 
 #  'Manchester United Football Ground': 0.13696978994957745,
 #  'Split 3': 0.11382061812590409,                       'Split 6': 0.09645682463644188,
 #  'Patricroft': 0.0875169123078903,                     'Manchester Oxford Road': 0.06277537107574177,
 #  'Trafford Park': 0.06181833052730487,                 'Split 8': 0.054673669286318755,
 #  'Swinton (Greater Manchester)': 0.05103723786885929,  'Clifton': 0.0426847965880539,
 #  'Manchester Piccadilly': 0.03005642560598291,         'Humphrey Park': 0.027869868541844808,
 #  'Split 9': 0.02468301017250274,                       'Split 5': 0.024648763510693426,
 #  'Moorside': 0.022272166324614012,                     'Split 25': 0.017360690371392252, 
 #  'Urmston': 0.012497072628777018,                      'Split 17': 0.011143908354095285, 
 #  'Split 4': 0.011052707611680656,                      'Walkden': 0.008352441280805386, 
 #  'Ardwick': 0.00828226793869966,                       'Split 10': 0.007937537255431186, 
 #  'Chassen Road': 0.005453705389415197,                 'Split 16': 0.0050320870340574975,
 #  'Split 7': 0.004823400194284327,                      'Split 12': 0.004710133056072124, 
 #  'Ashburys': 0.003792567865679234,                     'Split 18': 0.002273730716851716, 
 #  'Split 11': 0.002164451355161296,                     'Levenshulme': 0.0021337006834492925,
 #  'Mauldeth Road': 0.0021290135457600404,               'Flixton': 0.002045265630273547,
 #  'Moston': 0.0018088938081945226,                      'Split 21': 0.0010301398149274192,
 #  'Gorton': 0.0009851013824543164,                      'Belle Vue': 0.0009819185036577874, 
 #  'Heaton Chapel': 0.0009699431398354697,               'Burnage': 0.0009578342753422048,
 #  'Ashton-under-Lyne': 0.0004721441380176313,           'Fairfield': 0.00045437337814168343,
 #  'Split 15': 0.00044632486643911886,                   'Ryder Brow': 0.00044629488754453534,
 #  'East Didsbury': 0.0004195777768668631,               'Stalybridge': 0.00022737905450820385, 
 #  'Split 22': 0.0002208329739699211,                    'Stockport': 0.00021560166576195167, 
 #  'Reddish North': 0.00020336183430928848,              'Gatley': 0.00015787454116270657,
 #  'Split 29': 0.00013200651372178565,                   'Guide Bridge': 0.0001299517111115209, 
 #  'Split 35': 0.0001247735337300702,                    'Split 24': 0.00012137848520412086, 
 #  'Brinnington': 9.296195620435985e-05,                 'Split 34': 5.772712897293758e-05, 
 #  'Split 28': 5.678849171441926e-05,                    'Davenport': 5.556687651856205e-05, 
 #  'Bredbury': 4.2629501543909554e-05,                   'Split 30': 2.6861879223042095e-05, 
 #  'Split 32': 2.674709664903887e-05,                    'Woodsmoor': 2.121134045834938e-05, 
 #  'Split 14': 1.9530484572585668e-05,                   'Flowery Field': 1.2790779515107183e-05,
 #  'Navigation Road': 1.2108871570047767e-05,            'Romiley': 8.731206688287835e-06, 
 #  'Newton for Hyde': 5.963237924703261e-06,             'Altrincham': 4.687142062944129e-06,
 #  'Split 1': 3.3435575589179233e-06,                    'Godley': 2.3647451506522243e-06}
# =============================================================================





# For Manchester Victoria - Salford Central (original)
# nodes =  ['Split 8', 'Split 9', 'Split 2',
#            'Split 6','Deansgate', 'Split 13',  
#            'Split 33','Salford Central', 
#            'Manchester Victoria', 'Eccles', 'Patricroft',
#            'Manchester United Football Ground', 
#            'Trafford Park', 'Salford Crescent', 'Split 3',
#            'Clifton', 'Swinton (Greater Manchester)', 
#            'Moorside', 'Walkden']



# For Manchester Victoria - Salford Central (alternative route)
# nodes =  ['Split 8', 'Split 9', 'Split 2',
#             'Split 5',
#            'Split 6', 'Manchester Oxford Road',
#            'Deansgate', 'Manchester Piccadilly', 'Ashburys',
#            'Split 10', 'Split 11',  'Split 13', 
#            'Split 16', 'Split 17', 'Split 18', 
#            'Gorton', 'Fairfield', 
#             'Split 21', 'Split 22', 'Guide Bridge', 
#            'Split 24', 'Ashton-under-Lyne',  
#            'Stalybridge','Split 25', 'Split 33',
#            'Salford Central', 'Split 29', 
#            'Manchester Victoria']






 





























 





























 





























    
# def my_draw_networkx_edge_labels(
#     G,
#     pos,
#     edge_labels=None,
#     label_pos=0.5,
#     font_size=10,
#     font_color="k",
#     font_family="sans-serif",
#     font_weight="normal",
#     alpha=None,
#     bbox=None,
#     horizontalalignment="center",
#     verticalalignment="center",
#     ax=None,
#     rotate=True,
#     clip_on=True,
#     rad=0
# ):
#     """Draw edge labels.

#     Parameters
#     ----------
#     G : graph
#         A networkx graph

#     pos : dictionary
#         A dictionary with nodes as keys and positions as values.
#         Positions should be sequences of length 2.

#     edge_labels : dictionary (default={})
#         Edge labels in a dictionary of labels keyed by edge two-tuple.
#         Only labels for the keys in the dictionary are drawn.

#     label_pos : float (default=0.5)
#         Position of edge label along edge (0=head, 0.5=center, 1=tail)

#     font_size : int (default=10)
#         Font size for text labels

#     font_color : string (default='k' black)
#         Font color string

#     font_weight : string (default='normal')
#         Font weight

#     font_family : string (default='sans-serif')
#         Font family

#     alpha : float or None (default=None)
#         The text transparency

#     bbox : Matplotlib bbox, optional
#         Specify text box properties (e.g. shape, color etc.) for edge labels.
#         Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

#     horizontalalignment : string (default='center')
#         Horizontal alignment {'center', 'right', 'left'}

#     verticalalignment : string (default='center')
#         Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

#     ax : Matplotlib Axes object, optional
#         Draw the graph in the specified Matplotlib axes.

#     rotate : bool (deafult=True)
#         Rotate edge labels to lie parallel to edges

#     clip_on : bool (default=True)
#         Turn on clipping of edge labels at axis boundaries

#     Returns
#     -------
#     dict
#         `dict` of labels keyed by edge

#     Examples
#     --------
#     >>> G = nx.dodecahedral_graph()
#     >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

#     Also see the NetworkX drawing examples at
#     https://networkx.org/documentation/latest/auto_examples/index.html

#     See Also
#     --------
#     draw
#     draw_networkx
#     draw_networkx_nodes
#     draw_networkx_edges
#     draw_networkx_labels
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np

#     if ax is None:
#         ax = plt.gca()
#     if edge_labels is None:
#         labels = {(u, v): d for u, v, d in G.edges(data=True)}
#     else:
#         labels = edge_labels
#     text_items = {}
#     for (n1, n2), label in labels.items():
#         (x1, y1) = pos[n1]
#         (x2, y2) = pos[n2]
#         (x, y) = (
#             x1 * label_pos + x2 * (1.0 - label_pos),
#             y1 * label_pos + y2 * (1.0 - label_pos),
#         )
#         pos_1 = ax.transData.transform(np.array(pos[n1]))
#         pos_2 = ax.transData.transform(np.array(pos[n2]))
#         linear_mid = 0.5*pos_1 + 0.5*pos_2
#         d_pos = pos_2 - pos_1
#         rotation_matrix = np.array([(0,1), (-1,0)])
#         ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
#         ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
#         ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
#         bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
#         (x, y) = ax.transData.inverted().transform(bezier_mid)

#         if rotate:
#             # in degrees
#             angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
#             # make label orientation "right-side-up"
#             if angle > 90:
#                 angle -= 180
#             if angle < -90:
#                 angle += 180
#             # transform data coordinate angle to screen coordinate angle
#             xy = np.array((x, y))
#             trans_angle = ax.transData.transform_angles(
#                 np.array((angle,)), xy.reshape((1, 2))
#             )[0]
#         else:
#             trans_angle = 0.0
#         # use default box of white with white border
#         if bbox is None:
#             bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
#         if not isinstance(label, str):
#             label = str(label)  # this makes "1" and 1 labeled the same

#         t = ax.text(
#             x,
#             y,
#             label,
#             size=font_size,
#             color=font_color,
#             family=font_family,
#             weight=font_weight,
#             alpha=alpha,
#             horizontalalignment=horizontalalignment,
#             verticalalignment=verticalalignment,
#             rotation=trans_angle,
#             transform=ax.transData,
#             bbox=bbox,
#             zorder=1,
#             clip_on=clip_on,
#         )
#         text_items[(n1, n2)] = t

#     ax.tick_params(
#         axis="both",
#         which="both",
#         bottom=False,
#         left=False,
#         labelbottom=False,
#         labelleft=False,
#     )

#     return text_items