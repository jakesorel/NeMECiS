# Inference

This package takes the barcode alignments of the *Alignment* package and performs Bayesian inference to extract the 
posterior distribution of the mean log ZsGreen (and associated variance, as well as unsorted pool population density) 
for each synCRE. This is achieved in two steps. First sort bin gates are calibrated by analysis of purity-check flow 
cytometry data in *bounds_fitting_regularised* (details provided in Materials and Methods). Second, STAN is used for 
posterior distribution inference using the NUTS sampler. 

## Code
- [Bounds Fitting](inference/bounds_fitting_regularised.py): Using purity-check flow cytometry data on sorted populations, infer the sort-bin limits on a common scale of log ZsGreen. 
- [Run Stan](inference/run_stan.py): Python wrapper for NUTS posterior sampling of the log ZsGreen distribution and statistics of barcode proportions in the cell pool, cell sorting, and barcode amplification. This is performed on a by-synCRE, by-sample basis. 
- [Gate Model](inference/gate_model.stan): STAN code describing the probabilistic model of how an unsorted population of synCRE-integrated neural progenitors is (1) sorted; (2) sequenced. Models the raw count frequencies by sample. 