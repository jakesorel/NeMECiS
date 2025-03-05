##Now regularising by count matrix.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.mixture import GaussianMixture
import seaborn as sns
import jax
from jax import numpy as jnp
from jax import jit, jacrev,vmap,hessian
from scipy.optimize import curve_fit,minimize

jax.config.update('jax_enable_x64', True)
from joblib import delayed, Parallel


if not os.path.exists("bounds_fitting/plots"):
    os.mkdir("bounds_fitting/plots")

##Load FCS data for gates

fcs_files = []
for nm in os.listdir("reference/zsgreen_levels/"):
    if ".csv" in nm:
       fcs_files.append(nm)


rep = [1,2]
sag = [0,5]
quartile = [1,2,3,4]

zsgreen = np.empty((2, 2, 5), dtype=object)
for i, r in enumerate(rep):
    for j,s in enumerate(sag):
        for k, q in enumerate(quartile):
            dfi = pd.DataFrame()
            for fcs_file in fcs_files:
                if "%d%d%d"%(r,s,q) in fcs_file:
                    df = pd.read_csv("reference/zsgreen_levels/%s" % fcs_file)
                    dfi = pd.concat((dfi,df))
            zsgreen[i,j,k] = dfi["525_50 Blue-A"].values

for i, r in enumerate(rep):
    for j,s in enumerate([0,500]):
        dfi = pd.DataFrame()
        for fcs_file in fcs_files:
            if "r%d-%d" % (r, s) in fcs_file:
                print(r,s,fcs_file)
                df = pd.read_csv("reference/zsgreen_levels/%s" % fcs_file)
                dfi = pd.concat((dfi, df))
        zsgreen[i, j, 4] = dfi["525_50 Blue-A"].values

log_zsgreen = np.empty((2, 2, 5), dtype=object)
for i, R in enumerate(rep):
    for j, S in enumerate(sag):
        for k, Q in enumerate(quartile):
            zsg = zsgreen[i][j][k]
            log_zsgreen[i][j][k] = np.log(zsg[zsg > 0])
        zsg = zsgreen[i][j][4]
        log_zsgreen[i][j][4] = np.log(zsg[zsg > 0])

count_matrix = np.load("reference/count_matrix.npy")


##Fit gaussian mixture to each

##Calculate gaussian_mixture model for each gate

def extract_gmm_params(logzsg, n_components=3):
    """
    Fit using sklearn Gaussian Mixture Model for each fluorescence distribution individually
    """
    gm = GaussianMixture(n_components=n_components, random_state=0).fit(logzsg.reshape(-1, 1))
    mu = gm.means_.ravel()
    sigma = np.sqrt(gm.covariances_.ravel())
    w = gm.weights_.ravel()
    return w, mu, sigma

##Perform the fitting to extract the parameters.
mu_gate, sigma_gate, w_gate = np.zeros((2, 2, 5, 3)), np.zeros((2, 2, 5, 3)), np.zeros((2, 2, 5, 3))
for i, R in enumerate(rep):
    for j, S in enumerate(sag):
        for k in range(5):
            w_gate[i, j, k], mu_gate[i, j, k], sigma_gate[i, j, k] = extract_gmm_params(log_zsgreen[i][j][k])

@jit
def gaussian(x, mean, std):
    """General function of a gaussian"""
    return jnp.exp(-(x - mean) ** 2 / (2 * std ** 2)) / (std * jnp.sqrt(2 * jnp.pi))

@jit
def gaussian_mixture(x, ws, means, stds):
    """General generative function of a GMM"""
    pdf = jnp.zeros_like(x)
    for w, m, s in zip(ws, means, stds):
        pdf += w * gaussian(x, m, s)
    return pdf



@jit
def strong_cutoff(x,x0,x1,m):
    """0 to 1 to 0 function between 0 and 1, with smoothness m"""
    return 0.5*jnp.tanh((x-x0)*m)+ 0.5*jnp.tanh((x1-x)*m)


@jit
def make_cdf_kernel(x,x0,x1,alpha,m,gm):
    """
    Wrapping **strong_cuttof** with a weighting parameter alpha and gm
    gm is a vector of values of the same shape as x, providing a numerical approximation of the Gaussian Mixture Model
    """
    out = alpha*gm*strong_cutoff(x,x0,x1,m)
    return out

@jit
def get_mean_cdf_kernel(x,x0,x1,alpha,m,gm):
    """
    Integral of cdf in x.
    """
    cdf = make_cdf_kernel(x,x0,x1,alpha,m,gm)
    dx = x[1]-x[0]
    return cdf.sum()*dx


@jit
def cost_cdf_kernel(X,x,ys,gms,beta,qs,SAG_percent_matrix,gamma):
    """

    :param X: wrapper for the fit parameters (x0, x1, alpha,m).
        These are respectively
            x0: lower extent of the sort bound
            x1: upper extent of the sort bound
            alpha: scaler for the bound
            m: smoothness of the sort bound (proxy for experimental measurement error)
    :param x: vector of points (log fluorescence) used in the integral of the bounds calculation
    :param ys: list of probability distributions (enumerated over x) of each of the gates for each of the replicates (i.e. R1G1, R1G2, ... R1G4, R2G1, ...)
    :param gms: list of probability distributions (enumerated over x) of the corresponding global distribution. This repeats the value for replicate 1 4 times before repeating R2 four times.
    :param beta: regularisation parameter used to compare the estimated bound positions to the theoretical bound positions specified by percentiles.
    :param qs: Corresponding values of x theoretically expected for bound positions given percentiles of the global distribution
    :param SAG_percent_matrix: Frequency matrix of a set of synCREs for two replicates (shape 2 x 5 gates inc global x no.synCRE), used to regularise/batch control values across replicates.
    :param gamma: Corresponding regularisation parameter for the above.
    :return:
    """
    x0,x1,alpha,m = X[::4],X[1::4],X[2::4],X[3::4] ##extract the different pararameters (see above for description)
    dx = x[1]-x[0]
    cost = jnp.zeros((8))
    for k, (y,gm) in enumerate(zip(ys,gms)):
        cost = cost.at[k].add(((make_cdf_kernel(x, x0[k], x1[k], alpha[k], m[k], gm)-y)**2).sum()*dx) ##compare simulated vs true fluorescence distribution. This is a mean of the squared discrepancies in probabilities over space
        cost = cost.at[k].add(beta[0] * (qs[k][0] - x0[k]) ** 2 + beta[1] * (qs[k][1] - x1[k]) ** 2) ##compare the position of the lower and upper fit bounds against the percentile expectations, weighted by the regularisation parameter beta

    mean_gfps = jnp.zeros((8))
    for k, gm in enumerate(gms):
        mean_gfps = mean_gfps.at[k].set(get_mean_cdf_kernel(x, x0[k], x1[k], alpha[k], m[k], gm)) ##get mean fluourescence for each gate given the fit.
    mean_gfps = mean_gfps.reshape(2,4)

    gp = jnp.expand_dims(mean_gfps, 2) * SAG_percent_matrix[:, :4] ##mean gfp x frequencies of each synCRE
    approx_mean_gfp = gp.sum(axis=-2)/SAG_percent_matrix[:, :4].sum(axis=-2) ##normalise by frequencies, sum and divide to get an estimate of the mean fluourescence by synCRE
    dgfp_weighted_residual = jnp.sum(SAG_percent_matrix[:,4].mean(axis=0)*(approx_mean_gfp[1]-approx_mean_gfp[0])**2) ##compare the approximate mean fluorescence by synCRE across replicates, weighted by their relative frequencies and scaled by the regularisation parameter gamma
    return cost.sum()+dgfp_weighted_residual*gamma

def get_dgfp_weighted_residual(X,x,ys,gms,beta,qs,SAG_percent_matrix,gamma):
    ##see above for annotations.
    x0,x1,alpha,m = X[::4],X[1::4],X[2::4],X[3::4]
    mean_gfps = jnp.zeros((8))
    for k, gm in enumerate(gms):
        mean_gfps = mean_gfps.at[k].set(get_mean_cdf_kernel(x, x0[k], x1[k], alpha[k], m[k], gm))
    mean_gfps = mean_gfps.reshape(2,4)

    gp = jnp.expand_dims(mean_gfps, 2) * SAG_percent_matrix[:, :4]
    approx_mean_gfp = gp.sum(axis=-2)/SAG_percent_matrix[:, :4].sum(axis=-2)
    dgfp_weighted_residual = jnp.sum(SAG_percent_matrix[:,4].mean(axis=0)*(approx_mean_gfp[1]-approx_mean_gfp[0])**2)
    return dgfp_weighted_residual


jac = jit(jacrev(cost_cdf_kernel))

hess = jit(hessian(cost_cdf_kernel))


percent_matrix = count_matrix/np.expand_dims(count_matrix.sum(axis=-1),axis=-1)
percent_matrix_truncated = percent_matrix[...,~(percent_matrix[:,:,:4].sum(axis=-3)==0).any(axis=(0,1))]

def get_bounds_params(_beta,gamma,p_test):

    ##split data into test and train
    test_idx = np.random.choice(np.arange(percent_matrix_truncated.shape[-1]),int(percent_matrix_truncated.shape[-1]*p_test))
    train_idx = np.array(list(set(range(percent_matrix_truncated.shape[-1])).difference(set(list(test_idx)))))

    test_percent_matrix_truncated = percent_matrix_truncated[...,test_idx]
    train_percent_matrix_truncated = percent_matrix_truncated[...,train_idx]

    bound_params_all = np.zeros((2,2,4,4))
    dgfp_residuals_train = np.zeros(2)
    dgfp_residuals_test = np.zeros(2)

    ## Attempt to fit the gates
    for j in range(2): ##over SAG.

        ##Extract the gaussian mixture model fits for each of the fluorescence distributions
        r1_w_gateij, r1_mu_gateij, r1_sigma_gateij = w_gate[0, j], mu_gate[0, j], sigma_gate[0, j]
        r2_w_gateij, r2_mu_gateij, r2_sigma_gateij = w_gate[1, j], mu_gate[1, j], sigma_gate[1, j]


        ##Specify the range of fluorescence values over which to integrate
        x = np.linspace(3.5,13.5,200)

        ##Extract the GMM fits for the unsorted pool
        r1_w_p, r1_m_p, r1_s_p = r1_w_gateij[4], r1_mu_gateij[4], r1_sigma_gateij[4]
        r2_w_p, r2_m_p, r2_s_p = r2_w_gateij[4], r2_mu_gateij[4], r2_sigma_gateij[4]

        ##Evaluate GMM for the pool for each replicate.
        r1_gm = gaussian_mixture(x, r1_w_p, r1_m_p, r1_s_p)
        r2_gm = gaussian_mixture(x, r2_w_p, r2_m_p, r2_s_p)

        ##Evaluate the corresponding fluorescence values for the theoretical percentile points (i.e. 15% bounds with 13.33% spacings)
        percentile_points = jnp.array([0,15,15+40/3,15*2+40/3,15*2+2*40/3,15*3+2*40/3,15*3+40,100])
        r1_percentile_vals = jnp.array([np.percentile(log_zsgreen[0,j,4],p) for p in percentile_points])
        r1_percentile_vals_grid = r1_percentile_vals.reshape(-1,2)
        r2_percentile_vals = jnp.array([np.percentile(log_zsgreen[1,j,4],p) for p in percentile_points])
        r2_percentile_vals_grid = r2_percentile_vals.reshape(-1,2)
        qs = np.row_stack([r1_percentile_vals_grid,r2_percentile_vals_grid])

        #Evaluate GMM for each of the gates (and replicates)
        ys = []
        for i in range(2):
            for k in range(4):
                w, m, s = w_gate[i, j,k], mu_gate[i, j,k], sigma_gate[i, j,k]
                y = gaussian_mixture(x, w,m,s)
                ys.append(y)

        ##Wrap into a list the corresponding GMMs for each pool (note the repetition)
        gms = []
        for k in range(4):
            gms.append(r1_gm)
        for k in range(4):
            gms.append(r2_gm)

        ##Provide the initial conditions for the fit.
        ##Initialise x0 and x1 to the lower and upper bounds expected from the percentile calculation
        X0 = np.zeros((4*8))
        X0[::4] = np.concatenate((r1_percentile_vals_grid[:,0],r2_percentile_vals_grid[:,0]))
        X0[1::4] = np.concatenate((r1_percentile_vals_grid[:,1],r2_percentile_vals_grid[:,1]))
        X0[2::4] = 1.0
        X0[3::4] = 10.

        beta = np.array([_beta, _beta])

        res = minimize(cost_cdf_kernel, x0=X0, args=(x, ys, gms, beta, qs, train_percent_matrix_truncated[:,j], gamma), jac=jac, hess=hess,
                       method="Newton-CG")

        dgfp_residuals_train[j] = get_dgfp_weighted_residual(res.x,x, ys, gms, beta, qs, train_percent_matrix_truncated[:,j], gamma)
        dgfp_residuals_test[j] = get_dgfp_weighted_residual(res.x,x, ys, gms, beta, qs, test_percent_matrix_truncated[:,j], gamma)

        l = 0
        for i in range(2):
            for k in range(4):
                Xik = res.x.reshape(-1,4)[l]
                bound_params_all[i, j,k] = Xik
                l+=1
    return bound_params_all,dgfp_residuals_train,dgfp_residuals_test



_beta = 0.5623413251903491
gamma = 177.82794100389228


bound_params_all = np.zeros((2, 2, 4, 4))
dgfp_residuals_train = np.zeros(2)
dgfp_residuals_test = np.zeros(2)

## Attempt to fit the gates
for j in range(2):

    r1_w_gateij, r1_mu_gateij, r1_sigma_gateij = w_gate[0, j], mu_gate[0, j], sigma_gate[0, j]
    r2_w_gateij, r2_mu_gateij, r2_sigma_gateij = w_gate[1, j], mu_gate[1, j], sigma_gate[1, j]

    x = np.linspace(3.5, 13.5, 200)
    r1_count_matrix_ij = count_matrix[0, j]
    r2_count_matrix_ij = count_matrix[1, j]

    r1_w_p, r1_m_p, r1_s_p = r1_w_gateij[4], r1_mu_gateij[4], r1_sigma_gateij[4]
    r2_w_p, r2_m_p, r2_s_p = r2_w_gateij[4], r2_mu_gateij[4], r2_sigma_gateij[4]

    r1_gm = gaussian_mixture(x, r1_w_p, r1_m_p, r1_s_p)
    r2_gm = gaussian_mixture(x, r2_w_p, r2_m_p, r2_s_p)

    percentile_points = jnp.array(
        [0, 15, 15 + 40 / 3, 15 * 2 + 40 / 3, 15 * 2 + 2 * 40 / 3, 15 * 3 + 2 * 40 / 3, 15 * 3 + 40, 100])
    r1_percentile_vals = jnp.array([np.percentile(log_zsgreen[0, j, 4], p) for p in percentile_points])
    r1_percentile_vals_grid = r1_percentile_vals.reshape(-1, 2)
    r2_percentile_vals = jnp.array([np.percentile(log_zsgreen[1, j, 4], p) for p in percentile_points])
    r2_percentile_vals_grid = r2_percentile_vals.reshape(-1, 2)

    ys = []

    for i in range(2):
        for k in range(4):
            w, m, s = w_gate[i, j, k], mu_gate[i, j, k], sigma_gate[i, j, k]
            y = gaussian_mixture(x, w, m, s)
            ys.append(y)

    gms = []
    for k in range(4):
        gms.append(r1_gm)
    for k in range(4):
        gms.append(r2_gm)

    qs = np.row_stack([r1_percentile_vals_grid, r2_percentile_vals_grid])

    X0 = np.zeros((4 * 8))
    X0[::4] = np.concatenate((r1_percentile_vals_grid[:, 0], r2_percentile_vals_grid[:, 0]))
    X0[1::4] = np.concatenate((r1_percentile_vals_grid[:, 1], r2_percentile_vals_grid[:, 1]))
    X0[2::4] = 1.0
    X0[3::4] = 10.

    # gamma = 10
    # _beta = 10
    beta = np.array([_beta, _beta])

    res = minimize(cost_cdf_kernel, x0=X0, args=(x, ys, gms, beta, qs, percent_matrix_truncated[:, j], gamma),
                   jac=jac, hess=hess,
                   method="Newton-CG")

    dgfp_residuals_train[j] = get_dgfp_weighted_residual(res.x, x, ys, gms, beta, qs,
                                                         percent_matrix_truncated[:, j], gamma)
    dgfp_residuals_test[j] = get_dgfp_weighted_residual(res.x, x, ys, gms, beta, qs,
                                                        percent_matrix_truncated[:, j], gamma)

    l = 0
    for i in range(2):
        for k in range(4):
            Xik = res.x.reshape(-1, 4)[l]
            bound_params_all[i, j, k] = Xik
            l += 1

out_bounds = bound_params_all[:,:,:,:3].copy()
out_bounds[:,:,0,0] = -10. ##Reset the upper and lower bounds of the end states just to capture everythign.
out_bounds[:,:,3,1] = 20.

Zs = np.zeros((2,2,4))
for i in range(2):
    for j in range(2):
        for k in range(4):
            Zs[i,j,k] = ((log_zsgreen[i, j, 4] > out_bounds[i, j, k, 0]) * (log_zsgreen[i, j, 4] < out_bounds[i, j, k, 1])).mean()

out_bounds[:,:,:,2] = Zs

np.save("bounds_fitting/out_bounds.npy",out_bounds)


