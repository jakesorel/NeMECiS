import os
from cmdstanpy import cmdstan_path, CmdStanModel,install_cmdstan
import matplotlib.pyplot as plt
import numpy as np 
from scipy import stats
from scipy.interpolate import interp1d
import pandas as pd
import sys
# install_cmdstan()

if __name__ == "__main__":

    # specify locations of Stan program file and data
    stan_file = "gate_model.stan"
    
    # instantiate a model; compiles the Stan program by default
    model = CmdStanModel(stan_file=stan_file,exe_file="gate_model_2",force_compile=False)
    
    """
    Import data
    """
    
    
    filtered_counts_files = []
    for nm in os.listdir("../../alignment/filtered_counts"):
        if ".csv" in nm:
            filtered_counts_files.append("../../alignment/filtered_counts/"+nm)
    
    filtered_counts_files = sorted(filtered_counts_files)
    
    index_counts = []
    replicate = []
    sag = []
    quartile = []
    lane = []
    for fl in filtered_counts_files:
        name = fl.split("/")[-1].strip(".csv")
        rep,sg,q = name[0],name[1],name[2]
        ln = int(name.split("_")[2].strip("L00"))-1
        replicate.append(rep)
        sag.append(sg)
        quartile.append(q)
        lane.append(ln)
        dfi = pd.read_csv(fl)
        index_counts.append(dfi["Counts"])
    
    rep_dict = {"1":0,"2":1}
    sag_dict = {"0":0,"5":1}
    quartile_dict = {"1":0,"2":1,"3":2,"4":3,"A":4}
    
    count_matrix = np.zeros((2,2,5,25**3),dtype=int)
    for rep,sg,q,ln,idx_count, in zip(replicate,sag,quartile,lane,index_counts):
        i,j,k = rep_dict[rep],sag_dict[sg],quartile_dict[q]
        count_matrix[i,j,k] += idx_count
    
    count_matrix_tot = count_matrix.sum(axis=-1)

    df_sort = pd.read_csv("sorted_cells_statistics.csv")
    df_sort = df_sort.iloc[:16]
    N_sorted_cell = df_sort["Sorted Cells"].values.reshape(2,2,4).astype(int)

    out_bounds = np.load("out_bounds.npy")
    lbs = out_bounds[...,0].copy()
    lbs[lbs<0] = 1e-17
    ubs = out_bounds[...,1]
    Zs = out_bounds[...,2]

    def fit_with_stan(i,j,k):
        data = {"N_obs": 4, 
                "Y": list(count_matrix[i,j,:4,k]),
                "Q":count_matrix[i,j,4,k], 
                "N_cell": list(N_sorted_cell[i,j]), 
                "N_read": list(count_matrix_tot[i,j,:4]),
                "N_read_bulk": count_matrix_tot[i,j,4],
                "lb": list(lbs[i,j]), 
                "ub": list(ubs[i,j]), 
                "Z": list(Zs[i,j])}
    
        qs = np.array(data["Y"])/np.array(data["N_read"])
        pi_approx = data["Q"]/data["N_read_bulk"]
        mids = (np.array(data["lb"])+np.array(data["ub"]))/2
        mids[0] = data["ub"][0]
        mids[-1] = data["lb"][-1]
        approx_mu = (qs*mids).sum()/qs.sum()
        p_sorted_j_approx = list(qs*np.array(data["N_cell"]))
        n_chains = 8
        inits = []
        for c in range(n_chains):
            inits.append({"pi":pi_approx,
                          "mu":np.random.normal(approx_mu,0.5),
                          "sigma":np.random.uniform(0.8,1.5)})
        fit = model.sample(chains=n_chains, data=data,show_progress=False,iter_warmup=500,iter_sampling=500,inits=inits)
        summary = fit.summary()
        assert summary["R_hat"]["mu"]<1.05, "did not converge; try again"
        return summary
    
    def fit_with_stan_converge(i,j,k,max_attempts=10):
        result = None
        n_attempts = 0 
        while (result is None)*(n_attempts<=max_attempts):
            try:
                # connect
                result = fit_with_stan(i,j,k)
            except:
                n_attempts += 1
        if result is None:
            df_blank = pd.DataFrame(np.ones((8,9))*np.nan)
            df_blank.columns = ['Mean', 'MCSE', 'StdDev', '5%', '50%', '95%', 'N_Eff', 'N_Eff/s','R_hat']
            df_blank.index = ['lp__', 'mu', 'pi', 'sigma', 'p_sorted_j[1]', 'p_sorted_j[2]','p_sorted_j[3]', 'p_sorted_j[4]']
            result = df_blank
        return result


    slurm_index = int(sys.argv[1])

    N_to_sample = 25
    N_slurm_jobs = 25*25

    range_to_sample = np.arange(N_to_sample*slurm_index,N_to_sample*(1+slurm_index))

    
    for rep in range(2):
        for sag in range(2):
            summaries = []
            for k in range_to_sample:
                summary = fit_with_stan_converge(rep,sag,k)
                summary.to_csv("stan_results/all/%d_%d_%d.csv"%(rep,sag,k))
                summaries.append(summary)
            means = pd.DataFrame(np.array([summary["Mean"].values for summary in summaries]))
            means.columns = ["Mean_%s"%lab for lab in ['lp__', 'mu', 'pi', 'sigma', 'p_sorted_j[1]', 'p_sorted_j[2]','p_sorted_j[3]', 'p_sorted_j[4]']]
            means.index = range_to_sample
            meta = pd.DataFrame({"Rep":np.repeat(rep,len(range_to_sample)),
                                 "SAG":np.repeat(sag,len(range_to_sample)),
                                 "Id":range_to_sample})
            meta.index = range_to_sample
            std = pd.DataFrame(np.array([summary["StdDev"].values for summary in summaries]))
            std.columns = ["StdDev_%s"%lab for lab in ['lp__', 'mu', 'pi', 'sigma', 'p_sorted_j[1]', 'p_sorted_j[2]','p_sorted_j[3]', 'p_sorted_j[4]']]
            std.index = range_to_sample
            summary_df = pd.concat((meta,means,std),axis=1)
            summary_df.to_csv("stan_results/summary/%d_%d_batch_%d.csv"%(rep,sag,slurm_index))
            
            
            
                               
                               


