import pandas as pd
import numpy as np
from scipy.sparse import save_npz,csr_matrix
import os
from CREdential.model import Model
from sklearn.metrics import explained_variance_score,r2_score
from tqdm import tqdm

class HyperparameterTuning:
    """
    K-fold cross validation method to choose regularisation strength

    """
    def __init__(self,model=None,options=None):
        assert options is not None
        assert model is not None
        self.model = model
        self.options = options
        self.output = {}
        self.k_models = []
        self.initialize()

    def initialize(self):
        self.generate_equal_k_fold_split()
        self.repackage_input_by_split()
        self.perform_hyperparameter_scan()

    def generate_equal_k_fold_split(self):
        """
        Randomly select "split_choice_n_iter" k_fold splits. Select the split that shows the lowest squared difference in average abundance across features
        Choose "split seed" for reproducibility

        """
        n_iter = self.options["split_choice_n_iter"]
        np.random.seed(self.options["split_seed"])
        scores = np.zeros(n_iter)
        ids_splits = []
        ids_remainings = []

        av = np.array(self.model.input["Z"].mean(axis=0)).ravel()
        for i in range(n_iter):
            ids_split, ids_remaining = make_k_fold_split(self.model.input["Z"].shape[0], self.options["k"])
            scores[i] = ((get_feature_abundance(ids_remaining, self.model.input["Z"]) - av) ** 2).sum()
            ids_splits += [ids_split]
            ids_remainings += [ids_remaining]
        min_score_index = np.argmin(scores)
        ids_split = [np.array(list(v)) for v in ids_splits[min_score_index]]
        ids_remaining = [np.array(list(v)) for v in ids_remainings[min_score_index]]
        self.output["k_fold_splits"] = {}
        self.output["k_fold_splits"]["train_idx"] = ids_remaining
        self.output["k_fold_splits"]["test_idx"] = ids_split

    def repackage_input_by_split(self):
        """
        Generate a new folder "hyperparameter_k_fold" and within it generate numbered folders by each fold
        Generate new input folders for each to act as inputs for re-runs of the model, subsetting with the indices defined above.
        """
        mkdir(self.model.options["input_dir"] + "/hyperparameter_k_fold")

        df_meta = pd.DataFrame(self.model.input["Z_meta"])
        outs = {}
        for key in ["Y", 'weight']:
            outs[key] = self.model.input[key]
        data_meta = pd.DataFrame(outs)
        Z = self.model.input["Z"]

        for i, (test_ids,train_ids) in enumerate(zip(self.output["k_fold_splits"]["test_idx"],self.output["k_fold_splits"]["train_idx"])):
            input_dir = self.model.options["input_dir"] + "/hyperparameter_k_fold/%d"%i
            mkdir(input_dir)
            mkdir(input_dir+"/test")
            mkdir(input_dir+"/train")
            for ipd,idx in zip([input_dir+"/test",input_dir+"/train"],[test_ids,train_ids]):

                df_meta.to_csv(ipd + "/df_meta.csv")
                data_meta.iloc[idx].reset_index().to_csv(ipd + "/data_meta.csv")
                save_npz(ipd + "/Z.npz", Z[idx])

    def perform_hyperparameter_scan(self):
        """
        For each k_fold split, fit on the test, get prediction on the train, for varying alpha

        Compile the scores and save to csv files for records.

        save optimal alpha to inputs (argmax of explained_variance cost, nothing more fancy than that)
        """
        mkdir(self.model.options["results_dir"] + "/hyperparameter_k_fold")
        self.k_models = []
        scores = np.zeros((len(self.options["alpha_range"]),self.options["k"],2))
        for j, alpha in enumerate(tqdm(self.options["alpha_range"])):
            for i in range(self.options["k"]):
                _options = self.model.options.copy()
                _options["hyperparameters"]["alpha"] = alpha
                _options_train = _options.copy()
                _options_train["input_dir"] += "/hyperparameter_k_fold/%d/train"%i

                _options_test = _options.copy()
                _options_test["input_dir"] += "/hyperparameter_k_fold/%d/test"%i

                ##Fit on the train data
                mdl_train = Model(_options_train)
                mdl_train.fit()

                ##Collate the test data and make predictions
                mdl_test = Model(_options_test)
                Y_pred = mdl_train.predict(mdl_test.input["Z"])

                ##Calculate scores
                scores[j,i,0] = explained_variance_score(mdl_test.input["Y"], Y_pred)
                scores[j,i,1] = r2_score(mdl_test.input["Y"], Y_pred)

        ##Generate csv files of the scores
        df_expv = pd.DataFrame(scores[...,0])
        df_expv.index = ["log10alpha=%.3f"%np.log10(alpha) for alpha in self.options["alpha_range"]]
        df_expv.columns = ["Kf_%d"%i for i in range(self.options["k"])]
        df_expv.to_csv(self.model.options["results_dir"] + "/hyperparameter_k_fold/explained_variance_k_alpha.csv")

        df_r2 = pd.DataFrame(scores[...,1])
        df_r2.index = ["log10alpha=%.3f"%np.log10(alpha) for alpha in self.options["alpha_range"]]
        df_r2.columns = ["Kf_%d"%i for i in range(self.options["k"])]
        df_r2.to_csv(self.model.options["results_dir"] + "/hyperparameter_k_fold/r2_k_alpha.csv")

        ##Use the argmax of the explained variance score to get the optimal alpha
        self.output["opt_alpha"] = self.options["alpha_range"][np.argmax(scores[...,0].mean(axis=1))]



def mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def split_integer(num, parts):
    quotient, remainder = divmod(num, parts)
    lower_elements = [quotient for i in range(parts - remainder)]
    higher_elements = [quotient + 1 for j in range(remainder)]
    return lower_elements + higher_elements

def make_k_fold_split(n_indices, n_splits):
    ids = np.arange(n_indices)
    np.random.shuffle(ids)
    ids_split = []
    n_split_vals = split_integer(n_indices, n_splits)
    n = 0
    for i in n_split_vals:
        ids_split += [set(ids[n:n + i])]
        n += i
    ids_remaining = [set(ids).difference(ids_s) for ids_s in ids_split]
    return ids_split, ids_remaining

def get_feature_abundance(ids_remaining, Z):
    return np.array([Z[list(ids_r)].mean(axis=0).A.ravel() for ids_r in ids_remaining])


