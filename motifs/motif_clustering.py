import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

df_pairwise = pd.read_csv("reference/pairwise_comparisons.tab",sep="\t",header=1)

c_ids = np.concatenate([df_pairwise["#id1"].values,df_pairwise["id2"].values])
c_ids = np.unique(c_ids)
c_id_index = dict(zip(c_ids,np.arange(len(c_ids))))
c_id_index_inv = dict(zip(np.arange(len(c_ids)),c_ids))
jaspar_id_to_name = dict(zip(list(df_pairwise["#id1"])+list(df_pairwise["id2"]),[nm.capitalize() for nm in list(df_pairwise["name1"])+list(df_pairwise["name2"])]))


df_pairwise["_id1"] = [c_id_index[i] for i in df_pairwise["#id1"]]
df_pairwise["_id2"] = [c_id_index[i] for i in df_pairwise["id2"]]

Ncor = sparse.coo_matrix((df_pairwise["Ncor"].values,(df_pairwise["_id1"].values,df_pairwise["_id2"].values)),shape = (len(c_ids),len(c_ids)))

NcorA = Ncor.A
scores = np.zeros(200)
for i, n in enumerate(tqdm(range(100,300))):
    cluster = AgglomerativeClustering(n_clusters=n)
    cluster.fit(NcorA)

    labels = cluster.labels_
    unique_labels = np.unique(labels)
    average_internal_correlation = np.array([(NcorA[labels==i][:,labels==i]).mean() for i in unique_labels])
    average_external_correlation = np.array([(NcorA[labels==i][:,labels!=i]).mean() for i in unique_labels])
    correlation_score = (average_internal_correlation/average_external_correlation).mean()
    scores[i] = correlation_score


cluster = AgglomerativeClustering(n_clusters=150)
cluster.fit(NcorA)
labels = cluster.labels_
names = np.array([jaspar_id_to_name[c_id_index_inv[i]] for i in range(len(labels))])

pd.DataFrame({"name":names,"JASPAR":[c_id_index_inv[i] for i in range(len(labels))],"cluster":labels}).to_csv("scripts/custom_motif_clustering/archetypes_remade_test.csv")


arch = pd.read_csv("scripts/custom_motif_clustering/archetypes.csv")
