import importlib
import pandas as pd
from scipy.sparse import load_npz
from CREdential.utils import *
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, explained_variance_score

def get_model_class(model_name):
    name = model_name + "_model"
    class_name = "".join([nm.capitalize() for nm in name.split("_")])
    class_instance = getattr(importlib.import_module("CREdential.models.%s" % name), class_name)
    return class_instance

class Model:
    """
    Takes 'output' from ProcessFitData as input

    establishes model class
    """
    def __init__(self,options=None):
        assert options is not None
        self.options = options
        mkdir(self.options["results_dir"])
        self.input = {}
        self.load_input()
        self.model = get_model_class(self.options["model_name"])(self.input,self.options)
        self.output = {}
        self.bootstrap_models = []

    def load_input(self):
        data_meta = pd.read_csv(self.options["input_dir"] +"/data_meta.csv",index_col=0)
        self.input = dict(data_meta)
        for key, val in self.input.items():
            self.input[key] = val.values
        self.input["Z_meta"] = dict(pd.read_csv(self.options["input_dir"]+"/df_meta.csv",index_col=0))
        for key, val in self.input["Z_meta"].items():
            self.input["Z_meta"][key] = val.values
        self.input["Z"] = load_npz(self.options["input_dir"]+"/Z.npz")


    def fit(self):
        self.model.fit()
        self.output = self.model.out

    def predict(self,Z):
        return self.model.predict(Z)



    def generate_export_df(self):
        df_out = pd.DataFrame(self.input["Z_meta"])
        df_out["coef"] = self.output["feature_coef"]
        if "bootstrapped_feature_coef" in self.output:
            df_bootstrapped = pd.DataFrame(self.output["bootstrapped_feature_coef"]).T
            df_bootstrapped.columns = ["coef_boot%d"%i for i in range(len(self.output["bootstrapped_feature_coef"]))]
            df_bootstrapped["coef_boot_mean"] = df_bootstrapped.mean(axis=1)
            df_bootstrapped["coef_boot_std"] = df_bootstrapped.std(axis=1)
        df_out = pd.concat((df_out, df_bootstrapped), axis=1)
        df_out = pd.concat((df_out,pd.DataFrame({"feature_category":["intercept"],"SAG":[np.nan],"feature":["intercept"],"coef":self.output["model_coef"]})))
        df_out = df_out.reset_index()
        return df_out

    def export(self):
        self.generate_export_df().to_csv(self.options["results_dir"]+"/fit_coef.csv")

    def get_score(self):
        Y_pred = self.predict(self.input["Z"])
        Y_pred_0 = self.predict(self.input["Z"][self.input["Z"]@self.input["Z_meta"]["SAG"]==0])
        Y_pred_500 = self.predict(self.input["Z"][self.input["Z"]@self.input["Z_meta"]["SAG"]!=0])
        pearson_r = stats.linregress(self.input["Y"],Y_pred)[2]
        pearson_r_0 = stats.linregress(self.input["Y"][self.input["Z"]@self.input["Z_meta"]["SAG"]==0],Y_pred_0)[2]
        pearson_r_500 = stats.linregress(self.input["Y"][self.input["Z"]@self.input["Z_meta"]["SAG"]!=0],Y_pred_500)[2]

        r2 = r2_score(self.input["Y"],Y_pred)
        r2_0 = r2_score(self.input["Y"][self.input["Z"]@self.input["Z_meta"]["SAG"]==0],Y_pred_0)
        r2_500 = r2_score(self.input["Y"][self.input["Z"]@self.input["Z_meta"]["SAG"]!=0],Y_pred_500)

        expv = explained_variance_score(self.input["Y"],Y_pred)
        expv_0 = explained_variance_score(self.input["Y"][self.input["Z"]@self.input["Z_meta"]["SAG"]==0],Y_pred_0)
        expv_500 = explained_variance_score(self.input["Y"][self.input["Z"]@self.input["Z_meta"]["SAG"]!=0],Y_pred_500)

        score_dict = {"R":pearson_r,
                      "R_0":pearson_r_0,
                      "R_500":pearson_r_500,
                      "R2":r2,
                      "R2_0":r2_0,
                      "R2_500":r2_500,
                      "explained_variance":expv,
                      "explained_variance_0":expv_0,
                      "explained_variance_500":expv_500}

        score_df = pd.DataFrame(score_dict,index=["full"])
        if len(self.bootstrap_models)>0:
            pearson_rs = np.zeros((len(self.bootstrap_models)))
            pearson_rs_0 = np.zeros((len(self.bootstrap_models)))
            pearson_rs_500 = np.zeros((len(self.bootstrap_models)))

            r2s = np.zeros_like(pearson_rs)
            r2s_0 = np.zeros_like(pearson_rs)
            r2s_500 = np.zeros_like(pearson_rs)

            expvs = np.zeros_like(pearson_rs)
            expvs_0 = np.zeros_like(pearson_rs)
            expvs_500 = np.zeros_like(pearson_rs)

            for i in range(self.options["n_bootstrap"]):
                SAG_mask = self.bootstrap_models[i].input["Z"]@self.input["Z_meta"]["SAG"]!=0
                pearson_rs[i] = stats.linregress(self.bootstrap_models[i].input["Y"], self.bootstrap_models[i].predict(self.bootstrap_models[i].input["Z"]))[2]
                pearson_rs_0[i] = stats.linregress(self.bootstrap_models[i].input["Y"][~SAG_mask], self.bootstrap_models[i].predict(self.bootstrap_models[i].input["Z"])[~SAG_mask])[2]
                pearson_rs_500[i] = stats.linregress(self.bootstrap_models[i].input["Y"][SAG_mask], self.bootstrap_models[i].predict(self.bootstrap_models[i].input["Z"])[SAG_mask])[2]

                r2s[i] = r2_score(self.bootstrap_models[i].input["Y"], self.bootstrap_models[i].predict(self.bootstrap_models[i].input["Z"]))
                r2s_0[i] = r2_score(self.bootstrap_models[i].input["Y"][~SAG_mask], self.bootstrap_models[i].predict(self.bootstrap_models[i].input["Z"])[~SAG_mask])
                r2s_500[i] = r2_score(self.bootstrap_models[i].input["Y"][SAG_mask], self.bootstrap_models[i].predict(self.bootstrap_models[i].input["Z"])[SAG_mask])

                expvs[i] = explained_variance_score(self.bootstrap_models[i].input["Y"], self.bootstrap_models[i].predict(self.bootstrap_models[i].input["Z"]))
                expvs_0[i] = explained_variance_score(self.bootstrap_models[i].input["Y"][~SAG_mask], self.bootstrap_models[i].predict(self.bootstrap_models[i].input["Z"])[~SAG_mask])
                expvs_500[i] = explained_variance_score(self.bootstrap_models[i].input["Y"][SAG_mask], self.bootstrap_models[i].predict(self.bootstrap_models[i].input["Z"])[SAG_mask])

            score_dict_boot = {"R": pearson_rs,
                          "R_0": pearson_rs_0,
                          "R_500": pearson_rs_500,
                          "R2": r2s,
                          "R2_0": r2s_0,
                          "R2_500": r2s_500,
                          "explained_variance": expvs,
                          "explained_variance_0": expvs_0,
                          "explained_variance_500": expvs_500}
            score_df_boot = pd.DataFrame(score_dict_boot,index=["boot_%d"%i for i in range(self.options["n_bootstrap"])])
            score_df = pd.concat((score_df,score_df_boot))
        score_df.to_csv(self.options["results_dir"]+"/scores.csv")


    def bootstrap(self):
        """
        Bootstrapping is performed while holding the intercept fixed.
        :return:
        """
        np.random.seed(self.options["seed"])
        _inputs = []
        Z_sum = np.array(self.input["Z"].sum(axis=0)).ravel()
        _options = self.options.copy()
        _options["fit_intercept"]=False
        scores = []
        for i in range(self.options["n_bootstrap"]*100):
            new_idx = np.random.choice(np.arange(len(self.input["Y"])),len(self.input["Y"]))
            new_Z = self.input["Z"][new_idx]
            new_Z_sum = np.array(new_Z.sum(axis=0)).ravel()
            scores += [np.abs(np.log2(new_Z_sum)-np.log2(Z_sum)).max()]
            # accept_idx = (np.abs(np.log2(new_Z_sum)-np.log2(Z_sum))<self.options["min_2_fold_discrepancy"]).all()
            new_Y = self.input["Y"][new_idx]
            new_weight = self.input["weight"][new_idx]
            _inputs += [{"Y":new_Y-self.output["model_coef"][0],"weight":new_weight,"Z":new_Z}]

        scores = np.array(scores)
        chosen_idx = np.argsort(scores)[:self.options["n_bootstrap"]]
        if scores[chosen_idx].max() > self.options["min_2_fold_discrepancy"]:
            print("Max discrepancy score greater than settings; proceeding anyway. Tune the parameter")

        __inputs = []
        for i in chosen_idx:
            __inputs += [_inputs[i]]
        _inputs = __inputs

        self.bootstrap_models = [get_model_class(self.options["model_name"])(inpt,_options) for inpt in _inputs]
        [mdl.fit() for mdl in self.bootstrap_models]
        self.output["bootstrapped_feature_coef"] = np.array([mdl.out["feature_coef"] for mdl in self.bootstrap_models])

