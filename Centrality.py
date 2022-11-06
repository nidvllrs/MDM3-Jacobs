#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 13:59:38 2022

@author: tomrihoy
"""

import networkx as nx

#import graph network from 'bristol_network.py' with graph name 'G'
from bristol_network import G

def betweenness_func(G):

    betweenness_dict=nx.betweenness_centrality(G)
    max_betweenness=max(betweenness_dict, key=betweenness_dict.get)
    return max_betweenness


def eigenvector_func(G):
    
    eigenvector_dict=nx.eigenvector_centrality(G, max_iter=500)
    max_eigenvector=max(eigenvector_dict, key=eigenvector_dict.get)
    return max_eigenvector


print(eigenvector_func(G))
print(betweenness_func(G))
