import os
import sys
import multiprocessing

results_direc = "path_to_Fastq"
run_index = int(sys.argv[1])
if not os.path.exists("sam"):
    os.mkdir("sam")

results_files_rd1 = []
for nm in os.listdir(results_direc):
    if ".gz" in nm:
        if ".DS" not in nm:
            if "Undetermined" not in nm:
                rd = int(nm.split("_")[3].strip("R"))
                if rd == 1:
                    results_files_rd1.append(nm)
results_files_rd1 = sorted(results_files_rd1)
print(results_files_rd1,run_index)
index_name = ["_".join(nm.split("_")[:3]) for nm in results_files_rd1]
nm, id_nm = results_files_rd1[run_index],index_name[run_index]
if len(nm.split(".")) == 3:
    rd2_name = nm.split("_")
    rd2_name[3] = "R2"
    rd2_name = "_".join(rd2_name)
    rd1_name = nm
    os.system("bowtie2 --threads %d --trim-to 3:96 -x amplicons -1 %s -2 %s -S sam/%s.sam" % (multiprocessing.cpu_count(),
    results_direc + "/" + rd1_name, results_direc + "/" + rd2_name, id_nm))
    print(id_nm,"done")
