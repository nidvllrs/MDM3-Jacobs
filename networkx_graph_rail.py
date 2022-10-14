import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import momepy
import contextily as cx
from mpl_toolkits.basemap import Basemap

rail = gpd.read_file(r"C:\Users\ndvll\OneDrive\Bureau\rail_network.csv")
rail = rail.explode()
print(rail)
#rail_df = gpd.GeoDataFrame(rail,geometry=WKT)

rail.crs = "ETRS89"
print(rail['geometry'].crs)
graph = momepy.gdf_to_nx(rail, approach='dual')
#graph = momepy.node_degree(graph, name='degree')
#nx.draw(graph, node_size=15)
#rail.plot()
df_wm = rail.to_crs(epsg=4258)
print(df_wm.columns)
f, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
rail.plot(color='#e32e00', ax=ax[0])
for i, facet in enumerate(ax):
    facet.set_title(("Streets", "Dual graph", "Overlay")[i])
    facet.axis("off")
nx.draw(graph, {n:[n[0], n[1]] for n in list(graph.nodes)}, ax=ax[1], node_size=15)
rail.plot(color='#e32e00', ax=ax[2], zorder=-1)
nx.draw(graph, {n:[n[0], n[1]] for n in list(graph.nodes)}, ax=ax[2], node_size=15)
ax = df_wm.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
cx.add_basemap(ax, crs=df_wm.crs, source=cx.providers.OpenStreetMap.Mapnik)
plt.show()
