data {
    int<lower=0> N_obs;            // Number of observations
    array[N_obs] int<lower=0> Y;   // Observed values of Y_j
    array[N_obs] int<lower=0> N_cell;           // Number of cells
    array[N_obs] int<lower=0> N_read;           // Number of reads
    int<lower=0> Q;           // Number of reads in bulk
    int<lower=0> N_read_bulk;            // Number of reads of bulk
    array[N_obs] real<lower=0> lb;       // Lower bounds
    array[N_obs] real<lower=0> ub;       // Upper bounds
    array[N_obs] real<lower=0> Z;       // Pr in gate
}

parameters {
    real mu;                        // Mean parameter
    real<lower=0.00000001, upper=1> pi;      // Probability parameter
    real<lower=0.1> sigma;          // Standard deviation parameter
    array[N_obs] real<lower=0, upper=1> p_sorted_j;      // probability of sorting into each bin

}

model {
    // Priors
    mu ~ normal(6, 1);             // Prior for mu
    pi ~ beta(1, 1);                // Prior for pi
    sigma ~ scaled_inv_chi_square(1,0.1);        // Prior for sigma

    // Likelihood
    for (j in 1:N_obs) {
        // Define p_j
        real p_j = (pi * (Phi_approx((ub[j] - mu)/sigma) - Phi_approx((lb[j] - mu)/ sigma)) / Z[j]);

        // Get probability of sorted_j given N_cell[j]
        p_sorted_j[j] ~ normal((p_j),sqrt(p_j*(1-p_j)/N_cell[j])); //normal approximation to binomial Bin(N,p)/N. 

        // Get number of reads given number of cells 
        Y[j] ~ binomial(N_read[j], p_sorted_j[j]);
    }

    // Define Binomial likelihood for Q
    Q ~ binomial(N_read_bulk, pi);
}
