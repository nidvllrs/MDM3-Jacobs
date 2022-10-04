import csv
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
stops_df = pd.read_csv('Stops.csv')
passenger_num_2019 = pd.read_excel('rai0201.ods',engine="odf")
passenger_num_2019.drop([0,1,2,3,4,5,6,7],inplace=True)

print(passenger_num_2019.head())
# print(passenger_num_2019.columns)
# print(stops_df.head())
# print(stops_df.columns)
#
# G=nx.from_pandas_edgelist(stops_df, 'CommonName','LocalityName')
# nx.draw(G, with_labels=True, font_weight='bold')
# plt.savefig("filename.png")