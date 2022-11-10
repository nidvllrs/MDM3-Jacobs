import fiona
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon, shape, LineString

floods = gpd.read_file("path/to/Flood_Map_for_Planning_Rivers_and_Sea_Flood_Zone_3.shp")
#print(floods.schema)
stations = fiona.open("path/to/GB-railway-network-model/GB_rail_data/network/rail_nodes.shp")
edges = fiona.open("path/to/GB-railway-network-model/GB_rail_data/network/rail_edges.shp")
print(edges.schema)

df = pd.read_csv("path/to/nodes_2.csv")
#print(df.head())
def coord_lister(geom):
    coords = list(geom.exterior.coords)
    return (coords)
Multi = MultiPolygon([shape(pol['geometry']) for pol in fiona.open("path/to/data/Flood_Map_for_Planning_Rivers_and_Sea_Flood_Zone_3.shp")])
Polygons = list(Multi.geoms)
#print(Polygons)
coordinates_list = floods.geometry.apply(coord_lister)
stat_floods = []
#print(coordinates_list)
for x in range(len(df.index)):
    print(x)
    p=Point(df.iat[x,6],df.iat[x,7])
    for y in Polygons:
        if p.within(y) is True:
            stat_floods.append(p)
        else:
            pass


stat_floods_df = pd.DataFrame(stat_floods, columns=['points'])
stat_floods_df.to_csv("path/to/stats_floods.csv", encoding='utf-8')

#first = floods.next()
#print(first) # (GeoJSON format)
