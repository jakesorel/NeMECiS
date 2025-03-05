import pandas as pd
import numpy as np
from scipy import sparse
from scipy import stats
import json
import os

def format_ax(fig, ax):
    fig.subplots_adjust(bottom=0.3, left=0.3, right=0.8, top=0.8)
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(frameon=False)


with open('reference/aliases.json') as f:
    aliases = json.load(f)


##Load data

df_dense = pd.read_csv("reference/stan_values.csv", index_col=0)
threshold = 0.1
mask = (df_dense["G+_std"] < threshold).values * (df_dense["G-_std"] < threshold).values
mask *= df_dense["Pos2"]!=17
mask *= df_dense["Pos3"]!=16
_df_dense = df_dense[mask]

#### Load motifs
df = []
for fl in os.listdir("reference/jaspar/by_fragment"):
    if ".csv" in fl:
        dfi = pd.read_csv("reference/jaspar/by_fragment/" + fl, header=None)
        df += [dfi[np.arange(5)]]
df = pd.concat(df)
df["gene_name"] = [nm.split("__")[0].capitalize() for nm in df[1]]
df["jaspar"] = [nm.split("__")[1].replace(".pfm","") for nm in df[1]]
df = df[df[4] > 6.5]

jaspar_sorted = np.unique(df["jaspar"])
gene_name_sorted = np.unique(df["gene_name"])


jaspar_name_dict = dict(zip(jaspar_sorted, np.arange(jaspar_sorted.size)))
jaspar_name_dict_inv = dict(zip(np.arange(jaspar_sorted.size),jaspar_sorted))
jaspar_to_gene_name = dict(zip(df["jaspar"], df["gene_name"]))


jaspar_to_gene_adjacency = []
for j, g in zip(df["jaspar"],df["gene_name"]):
    jaspar_to_gene_adjacency += [tuple((list(jaspar_sorted).index(j),list(gene_name_sorted).index(g)))]

jaspar_to_gene_adjacency = np.array(jaspar_to_gene_adjacency)
jaspar_to_gene_adjacency_matrix = sparse.coo_matrix(([True]*len(jaspar_to_gene_adjacency),(jaspar_to_gene_adjacency[:,0],jaspar_to_gene_adjacency[:,1])))


##fragment  adjacency
fragment_motif_adjacency = []
for frag, jaspar_name in zip(df[0], df["jaspar"]):
    fragment_motif_adjacency += [tuple((int(frag.split("_")[1]), jaspar_name_dict[jaspar_name]))]
fragment_motif_adjacency = np.array(fragment_motif_adjacency)

fragment_motif_adjacency_matrix = sparse.coo_matrix((np.ones(fragment_motif_adjacency.shape[0], dtype=bool),
                                                     (fragment_motif_adjacency[:, 0], fragment_motif_adjacency[:, 1])))
fragment_motif_adjacency_matrix = fragment_motif_adjacency_matrix.A


##Extract expressed genes

df_RNA = pd.read_csv("reference/RNA_normCounts_filter1.csv", index_col=0)
df_RNA = df_RNA[list(df_RNA.columns)[3:]]
df_RNA.index = [nm.capitalize() for nm in df_RNA.index]

min_expr = 20.
expressed_genes = [nm.capitalize() for nm in df_RNA.index[df_RNA.sum(axis=1) > min_expr]]

is_expressed = []
expressed_motifs = []
not_expressed = []
for j in jaspar_sorted:
    _nm = jaspar_to_gene_name[j]
    if "::" in _nm:
        nms = _nm.split("::")
        nms = [n.capitalize() for n in nms]
    else:
        nms = [_nm]
    is_e = 0
    for nm in nms:
        if nm.capitalize() in expressed_genes:
            is_e += 1
        else:
            if nm.capitalize() in aliases:
                if aliases[nm.capitalize()] in expressed_genes:
                    is_e += 1
    if is_e == len(nms): ##i.e. for dimers, require both to be present
        is_expressed += [1]
        expressed_motifs += [_nm]
    else:
        is_expressed += [0]
        not_expressed += [_nm]

is_expressed = np.array(is_expressed) ##by jaspar


###Get archetypes

df_archetype = pd.read_csv("scripts/custom_motif_clustering/archetypes.csv", index_col=0)
df_archetype["jaspar_name"] = [nm.split("_")[-2].replace(".","-") for nm in df_archetype["JASPAR"]]

jaspar_to_archetype = dict(zip(df_archetype["jaspar_name"], df_archetype["cluster"]))


motif_id_archetype = []
for nm, c in jaspar_to_archetype.items():
    if nm in jaspar_to_gene_name:
        motif_id_archetype += [[c,jaspar_name_dict[nm]]]

motif_id_archetype = np.array(motif_id_archetype)
motif_to_archetype = sparse.coo_matrix(([True]*len(motif_id_archetype),(motif_id_archetype[:,0],motif_id_archetype[:,1])))
archetype_expressed = (motif_to_archetype@is_expressed)!=0


def calculate_spearman(_df_dense):
    pos_index = _df_dense[["Pos1", "Pos2", "Pos3"]]
    dGFP = _df_dense["G+"] - _df_dense["G-"]

    frag_in_data = np.zeros((len(_df_dense), 25), dtype=bool)
    for i in range(25):
        frag_in_data[:, i] = (pos_index == i).any(axis=1)

    archetype_in_data = frag_in_data @ ((motif_to_archetype@fragment_motif_adjacency_matrix.T).T)

    archetype_stats = np.zeros((archetype_in_data.shape[1], 2))
    for i in range(archetype_in_data.shape[1]):
        archetype_stats[i] = stats.spearmanr(archetype_in_data[:, i], dGFP)

    return archetype_stats

df_dense_r1 = _df_dense[_df_dense["Rep"]==0]
df_dense_r2 = _df_dense[_df_dense["Rep"]==1]

archetype_stats = np.array([calculate_spearman(df_dense_r1),calculate_spearman(df_dense_r2)])
mean_archetype_stats = archetype_stats.mean(axis=0)
