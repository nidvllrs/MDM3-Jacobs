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


G.add_edges_from([(S3,S5),(S5,S12),(S12,S6),(S6,S8),(S8,S4),(S4,S9),(S9,S10),(S10,S1),(S1,S11),(S3,S2),(S2,S7),
                  (S12,S13)])
G.add_edges_from([(S5,S3),(S12,S5),(S6,S12),(S8,S6),(S4,S8),(S9,S4),(S10,S9),(S1,S10),(S11,S1),(S2,S3),(S7,S2),
                  (S13,S12)])

node_colour=['b','b','r','b','b','b','b','b','b','b','b','b']
nx.draw(G, with_labels=True)

plt.draw()
plt.show()
