import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import momepy
from mpl_toolkits.basemap import Basemap
rail = gpd.read_file(r"C:\Users\ndvll\OneDrive\Bureau\rail_network.csv")

rail  = rail.explode()
graph = momepy.gdf_to_nx(rail, approach='dual')
graph = momepy.node_degree(graph, name='degree')
#nx.draw(graph, node_size=15)
#rail.plot()

nodes, edges, sw = momepy.nx_to_gdf(graph, points=True, lines=True,
                                    spatial_weights=True)
f, ax = plt.subplots(figsize=(10, 10))
nodes.plot(ax=ax, column='degree', cmap='tab20b', markersize=(nodes['degree'] * 100), zorder=2)
edges.plot(ax=ax, color='lightgrey', zorder=1)
ax.set_axis_off()
plt.show()
