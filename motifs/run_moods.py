import os
from joblib import Parallel, delayed

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

mkdir("alignment")
mkdir("alignment/by_fragment")


##Run MOODS on fragments:

file_list = []
for fl in [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser("fasta")) for f in fn]:
    if ".fa" in fl:
        file_list.append(fl.replace(".fa","").replace("fasta/",""))


file_list_fragments = []
for fl in file_list:
    if "by_fragment" in fl:
        file_list_fragments.append(fl)


def run_moods(nm):
    os.system("python MOODS/python/scripts/moods-dna.py -m reference/JASPAR2024_CORE_PROCESSED/*.pfm "
              "-s fasta/%s.fa -p 0.0001 "
              "> alignment/%s.csv"%(nm,nm))

Parallel(n_jobs=-1,backend="loky", prefer="threads")(delayed(run_moods)(nm) for nm in file_list)

Parallel(n_jobs=-1,backend="loky", prefer="threads")(delayed(run_moods)(nm) for nm in file_list_fragments)
