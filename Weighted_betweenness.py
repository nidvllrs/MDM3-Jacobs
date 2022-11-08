#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 14:34:24 2022

@author: tomrihoy
"""

import numpy as np
from bristol_network import G
import networkx as nx
from numpy import unravel_index


np.random.seed(seed=11)
Sta_num=len(list(G))
#random OD matrix-needs updating with actual values
ODmat=np.random.randint(2, size=(Sta_num,Sta_num))
Pathmat=np.empty((Sta_num,Sta_num),dtype=object)
Pathmat_num=np.empty((Sta_num,Sta_num),dtype=object)
Cent_mat=np.zeros((Sta_num,Sta_num))
ADJmat=nx.to_numpy_array(G)


#create dictionary with station names as keys, and numbers as values. This is 
#as the number of the stations will be used to input values into matrices that 
#represent various relationships between the stations
dicts={}
values=list(G)
for i in range(len(list(G))):
    dicts[i]=values[i]
dictionary_rev = {v: k for k, v in dicts.items()}




#convert path list of stations to their dictionary encoded number to give list
#of staion numbers
def convert_to_num(x):
    k=0
    holding_list=[]
    while k <= len(x)-1:
        holding_list.append(dictionary_rev[''.join(x[k])])
        k+=1
    return holding_list


#count the number of times an edge is crossed via the encoded(with numbers 
#via dictionary_rev) shortest paths between all stations
def count_edges(x):
    Path_length=len(x)
    i=0
    j=1
    if Path_length!=1:
        while j<Path_length-1:
            row=x[i]
            column=x[j]
            Cent_mat[row,column]=Cent_mat[row,column]+1
            i+=1
            j+=1
        
    
#find shortest path between all stations       
for i in range(Sta_num):
    for j in range(Sta_num):
        Pathmat[i,j]=nx.dijkstra_path(G, list(G)[i], list(G)[j])
        Pathmat_num[i,j]=convert_to_num(Pathmat[i,j])
        
#count how many times an edge is crossed in all shortest paths
a=0
b=0
for a in range(len(Pathmat_num)):
    for b in range(len(Pathmat_num)):
        count_edges(Pathmat_num[a,b])  
        
        
Cent_mat=np.reshape(Cent_mat, (Sta_num,Sta_num))
Cent_mat=Cent_mat/ODmat.sum()
Cent_mat=np.multiply(ADJmat,Cent_mat)


Max_val=unravel_index(Cent_mat.argmax(), Cent_mat.shape)





        









        





