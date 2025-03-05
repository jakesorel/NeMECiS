from CREdential.assembler import Assembler
from CREdential.process_fit_data import ProcessFitData
from CREdential.model import Model
from CREdential.hyperparameter_tuning import HyperparameterTuning
from CREdential.utils import *
import numpy as np


if __name__ == "__main__":


    mkdir("results_synergy")
    for rep, lab in enumerate(["results_r1","results_r2"]):
        results_dir = "results_synergy/%s"%lab
        mkdir(results_dir)

        assemble_options = {"expression_data":"reference/stan_values.csv",
                   "replicate":rep,
                   "drop_indices":[[1,17],[2,16]], ##[position,index]
                   "std_threshold":0.1,
                   "export_file_name":None
                   }
        process_data_options =  {"modelled_features":['fragments', 'fragment_pairs',"fragment_pair_relative_positions"],#'fragments', 'fragment_pairs',
                   "weight":True,
                                 "results_dir":results_dir}

        log_mult_model_options = {"model_name": "log_multiplicative",
                                  "hyperparameters": {"alpha": 0.01},
                                 "results_dir":results_dir + "/log_multiplicative",
                                  "input_dir":results_dir + "/input",
                                  "fit_intercept":True,
                                  "seed":2024,
                                  "min_2_fold_discrepancy":1,
                                  "n_bootstrap":50}

        hyperparameter_tuning_options = {"split_choice_n_iter": 100, "split_seed": 2024,"k":5,
                                         "alpha_range":np.logspace(-2,2,32)}

        assemble = Assembler(assemble_options)
        processed_data = ProcessFitData(assemble.input,process_data_options)

        ##log mult model
        log_mult_model = Model(log_mult_model_options)

        log_mult_hyperparameter_tuning = HyperparameterTuning(log_mult_model,hyperparameter_tuning_options)

        log_mult_model_options_optimised = log_mult_model_options.copy()
        log_mult_model_options_optimised["hyperparameters"]["alpha"] = log_mult_hyperparameter_tuning.output["opt_alpha"]
        log_mult_model_refit = Model(log_mult_model_options_optimised)
        log_mult_model_refit.fit()
        log_mult_model_refit.bootstrap()
        log_mult_model_refit.get_score()
        log_mult_model_refit.export()

