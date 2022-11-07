import networkx as nx
import matplotlib.pyplot as plt
G=nx.DiGraph()

S1='Avonmouth'
S2='Bedminster'
S3='Bristol TM'
S4='Clifton Down'
S5='Lawrence Hill'
S6='Montpelier'
S7='Parson Street'
S8='Redland'
S9='Sea Mills'
S10='Shirehampton'
S11='St Andrews'
S12='Stapleton Road'
S13='Filton Abbey Wood'

Station_array=[S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12]

G.add_nodes_from(Station_array)

G.add_weighted_edges_from([(S3,S5,1),(S5,S12,1),(S12,S6,1),(S6,S8,1),(S8,S4,1),(S4,S9,1),(S9,S10,1),(S10,S1,1),(S1,S11,1),(S3,S2,1),(S2,S7,1),(S12,S13,1)])


G.add_weighted_edges_from([(S5,S3,1),(S12,S5,1),(S6,S12,1),(S8,S6,1),(S4,S8,1),(S9,S4,1),(S10,S9,1),(S1,S10,1),(S11,S1,1),(S2,S3,1),(S7,S2,1),(S13,S12,1)])



node_colour=['b','b','r','b','b','b','b','b','b','b','b','b']
nx.draw(G, with_labels=True)

plt.draw()
plt.show()
