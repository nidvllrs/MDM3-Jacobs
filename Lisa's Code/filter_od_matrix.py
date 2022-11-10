import pandas as pd

df = pd.read_csv("path/to/GB-railway-network-model/GB_rail_data/outputs/od_matrix.csv")
df2 = pd.read_csv("path/to/bristol_ids.csv",sep=';')

list = []

for i in range(len(df2)):
    print(i)
    x = df2["node_id"][i]
    for j in range(len(df)):
        y = df["node_path"][j]
        if x in y:
            list.append(df.iloc[j])
        else:
            pass

df3 = pd.DataFrame(list,columns=["number","origin_id","destination_id","node_path","edge_path","distance","journeys"]  )
df3.to_csv("path/to/bristol_od_matrix.csv")
