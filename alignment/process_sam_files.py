import os
import sys
import numpy as np
import pandas as pd

run_index = int(sys.argv[1])

def mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

mkdir("processed_sam")

mkdir("filtered_counts")

sam_files = []
for fl in os.listdir("sam"):
    if ".DS" not in fl:
        sam_files.append(fl)

def get_index_from_triple(triple):
    ints = np.array(triple.split("-")).astype(int)-1
    return (ints*np.array((1,25,625))).sum()



fl_nm = sam_files[run_index]
name = fl_nm.strip(".sam")
fl = open("sam/%s"%fl_nm, "r").readlines()[25 ** 3 + 2:]
fl = [ln.split("\t") for ln in fl]
flags = np.array([int(ln[1]) for ln in fl])
flag_mask = flags == 99
aligned_seq = np.array([ln[2] for ln in fl])[flag_mask]
quality_score = np.array([int(ln[4]) for ln in fl])[flag_mask]
start_index = np.array([int(ln[8]) for ln in fl])[flag_mask]
cigar = np.array([ln[5] for ln in fl])[flag_mask]

df = pd.DataFrame({"Aligned":aligned_seq,"Quality":quality_score,"Start":start_index,"CIGAR":cigar})
df.to_csv("processed_sam/%s.csv"%name)

aligned_seq,quality_score,cigar = df["Aligned"],df["Quality"],df["CIGAR"]

K,J,I = np.mgrid[:25,:25,:25]
K,J,I = K.ravel()+1,J.ravel()+1,I.ravel()+1


filtered_seqs = aligned_seq[(aligned_seq!="*")*(cigar=="96M")]
index = list(map(get_index_from_triple,filtered_seqs))
index_counts = np.bincount(index,minlength=25**3)

df_filtered = pd.DataFrame({"Index":np.arange(25**3),"Pos1":I,"Pos2":J,"Pos3":K,"Counts":index_counts})
df_filtered.to_csv("filtered_counts/%s.csv"%name)

