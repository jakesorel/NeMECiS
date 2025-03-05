import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
import os
from CREdential.utils import *


class ProcessFitData:
    """
    Converts 'input' from Assemble into a minimalistic description of the
    feature matrix Z with annotations

    And calculates the weights
    """
    def __init__(self,input=None,options=None):
        assert input is not None
        assert options is not None
        self.input = input
        self.options=options
        self.output = {}
        self.get_output()


    def get_output(self):
        self.generate_feature_matrix()
        self.get_Y()
        self.assign_weights()
        self.export()

    def generate_feature_matrix(self):
        """
        Z is the feature matrix (n_data x (2*n_feature))
        Generate a corresponding metadata dictionary to keep track of features, each vectors of shape (2 * n_feature)

        """
        Z = np.column_stack([self.input["feature_count_SAG"][key] for key in self.options["modelled_features"]])
        Z_meta = {}
        Z_meta["feature_category"] = np.concatenate([np.repeat(key,self.input["feature_count_SAG"][key].shape[1]) for key in self.options["modelled_features"]])
        Z_meta["SAG"] = np.concatenate([np.repeat([0,1],int(self.input["feature_count_SAG"][key].shape[1]/2)) for key in self.options["modelled_features"]])
        feature = []
        for key in self.options["modelled_features"]:
            f = self.input["features"][key]
            if f.ndim == 1:
                feature += [np.tile(f.astype(str),2)]
            else:
                feature += [np.tile(["_".join(entry) for entry in list(f.astype(str))],2)]

        Z_meta["feature"] = np.concatenate(feature)
        self.output["Z"] = Z
        self.output["Z_meta"] = Z_meta

    def get_Y(self):
        self.output["Y"] = self.input["data"]["Y"]


    def assign_weights(self):
        self.output["weight"] = np.ones_like(self.input["data"]["Y_std"])
        if self.options["weight"]:
            self.output["weight"] = 1/self.input["data"]["Y_std"]**2
        self.output["weight"] /= self.output["weight"].mean()

    def export(self):
        df_meta = pd.DataFrame(self.output["Z_meta"])
        outs = {}
        for key in ["Y",'weight']:
            outs[key] = self.output[key]
        data_meta = pd.DataFrame(outs)
        mkdir(self.options["results_dir"]+"/input")
        df_meta.to_csv(self.options["results_dir"]+"/input/df_meta.csv")
        data_meta.to_csv(self.options["results_dir"]+"/input/data_meta.csv")
        save_npz(self.options["results_dir"]+"/input/Z.npz",csr_matrix(self.output["Z"]))


