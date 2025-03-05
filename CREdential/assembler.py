import numpy as np
import pandas as pd
import math
from scipy.sparse import csr_matrix
from itertools import combinations, permutations, combinations_with_replacement,product
import pickle

class Assembler:
    def __init__(self,options=None):
        assert options is not None
        self.options = options
        self.input = {}
        self.input["assemble_options"]=options
        self.run()

    def run(self):
        self.load_data()
        self.get_param_dimensions()
        self.get_features()
        self.extract_features()
        self.export()

    def load_data(self):
        df = pd.read_csv(self.options["expression_data"], index_col=0)
        df_rep = df[df["Rep"]==self.options["replicate"]]
        pos_ids = df_rep[["Pos1", "Pos2", "Pos3"]].values

        reject_mask = np.zeros(len(pos_ids),dtype=bool)
        for (i,j) in self.options["drop_indices"]:
            reject_mask += (pos_ids[:, i] == j)
        df_rep = df_rep[~(reject_mask)]

        self.input["data"] = {}
        X = df_rep[["Pos1", "Pos2", "Pos3"]].values
        _X = np.row_stack([X,X])
        S = np.repeat([0,1],len(X))
        Y = df_rep[['G-','G+']].values.T.ravel()
        Y_std = df_rep[['G-_std','G+_std']].values.T.ravel()


        std_threshold_mask = Y_std<=self.options["std_threshold"]
        for lab, val in zip(["X","S","Y","Y_std"],[_X,S,Y,Y_std]):
            self.input["data"][lab] = val[std_threshold_mask]

        n_data = len(self.input["data"]["X"])
        self.input["data"]["n_data"] = n_data

    def get_param_dimensions(self):
        """
        calculate the dimensions of each of the classes of parameters
        """
        dims = {}
        dims["n_positions"] = self.input["data"]["X"].shape[1]
        dims["fragment_ids"] = np.unique(self.input["data"]["X"].ravel())
        dims["n_fragments"] = len(dims["fragment_ids"])
        dims["n_fragments_positions"] = dims["n_fragments"]*dims["n_positions"]
        dims["n_fragment_pairs"] = int((dims["n_fragments"]*(dims["n_fragments"]+1))/2)
        dims["n_pair_positions"] = math.perm(dims["n_positions"],2)
        dims["n_pair_unordered_positions"] = math.comb(dims["n_positions"],2)
        dims["n_pair_relative_positions"] = np.unique(np.diff(np.array(list(permutations(np.arange(dims["n_positions"]),2))))).size
        dims["n_pair_unordered_relative_positions"] = np.unique(np.diff(np.array(list(combinations(np.arange(dims["n_positions"]),2))))).size

        dims["n_fragment_pair_positions"] = int((dims["n_fragment_pairs"]-dims["n_fragments"])*dims["n_pair_positions"] + dims["n_fragments"]*dims["n_pair_unordered_positions"])
        dims["n_fragment_pair_relative_positions"] = int((dims["n_fragment_pairs"]-dims["n_fragments"])*dims["n_pair_relative_positions"] + dims["n_fragments"]*dims["n_pair_unordered_relative_positions"])

        self.input["dims"] = dims

    def get_features(self):
        """
        Populates dictionary 'features' with indices

        fragments = 1D vector of ids
        fragment_pairs 2D matrix of (id,id)
        fragment_positions = 2D vector of (id, pos)
        pair_positions = 2D vector of (pos,pos)
        fragment_pair_positions = 2D vector of (id,id,pos,pos)
        fragment_pair_rel_positions = 2D vector of (id,id,displacement (pos2-pos1))
        """
        features = {}
        features["fragments"] = np.arange(self.input["dims"]["n_fragments"])
        features["fragment_pairs"] = np.array(list(combinations_with_replacement(features["fragments"],2)))
        features["fragment_position"] = np.array(list(product(features["fragments"],np.arange(self.input["dims"]["n_positions"]))))
        features["pair_positions"] = np.array(list(permutations(np.arange(self.input["dims"]["n_positions"]),2)))
        features["pair_unordered_positions"] = np.array(list(combinations(np.arange(self.input["dims"]["n_positions"]),2)))
        features["pair_relative_positions"] = np.unique(features["pair_positions"][:,1]-features["pair_positions"][:,0])
        features["pair_unordered_relative_positions"] = np.unique(features["pair_unordered_positions"][:,1]-features["pair_unordered_positions"][:,0])

        features["fragment_pair_positions"] = []
        for pair in features["fragment_pairs"]:
            if pair[0] != pair[1]:
                for pos12 in features["pair_positions"]:
                    features["fragment_pair_positions"] += [[pair[0],pair[1],pos12[0],pos12[1]]]
            else:
                for pos12 in features["pair_unordered_positions"]:
                    features["fragment_pair_positions"] += [[pair[0],pair[1],pos12[0],pos12[1]]]
        features["fragment_pair_positions"] = np.array(features["fragment_pair_positions"])

        features["fragment_pair_relative_positions"] = []
        for pair in features["fragment_pairs"]:
            if pair[0] != pair[1]:
                for pos12 in features["pair_relative_positions"]:
                    features["fragment_pair_relative_positions"] += [[pair[0], pair[1], pos12]]
            else:
                for pos12 in features["pair_unordered_relative_positions"]:
                    features["fragment_pair_relative_positions"] += [[pair[0], pair[1], pos12]]
        features["fragment_pair_relative_positions"] = np.array(features["fragment_pair_relative_positions"])


        self.input["features"] = features

    def extract_features(self):
        """
        For each datapoint (synCRE x SAG, ravelled) determine the count of each feature, while guarding against double counting
        """
        data_feature_count = {}

        ##Count fragments and fragment positions
        data_feature_count["fragment_position"] = (self.input["features"]["fragments"] == np.expand_dims(self.input["data"]["X"],2)).transpose(0,2,1).astype(int)
        data_feature_count["fragments"] = data_feature_count["fragment_position"].sum(axis=2)
        data_feature_count["fragment_position"] = data_feature_count["fragment_position"].reshape(self.input["data"]["n_data"],-1)

        #Count fragment x fragment  x their positions
        fragment_pairs_positions_count = np.zeros((self.input["data"]["n_data"],self.input["dims"]["n_fragment_pair_positions"]),dtype=int)
        for j, (f1,f2,pos1,pos2) in enumerate(self.input["features"]["fragment_pair_positions"]):
            mask = self.input["data"]["X"][:,[pos1,pos2]] == (f1,f2)
            mask = mask.all(axis=-1)
            fragment_pairs_positions_count[:,j] = mask

        data_feature_count["fragment_pair_positions"] = fragment_pairs_positions_count

        ##Determine a conversion matrix of pairs x position to pairs
        pair_positions_to_pair = np.zeros((self.input["dims"]["n_fragment_pair_positions"],self.input["dims"]["n_fragment_pairs"]),dtype=int)
        for j, (f1,f2) in enumerate(self.input["features"]["fragment_pairs"]):
            pair_positions_to_pair[:,j] = ((self.input["features"]["fragment_pair_positions"][:, [0, 1]] == [f1, f2]).all(axis=1))

        ##Determine a conversion matrix of pairs x position to pairs x relative position
        pair_positions_to_relative_positions = np.zeros((self.input["dims"]["n_fragment_pair_positions"],self.input["dims"]["n_fragment_pair_relative_positions"]),dtype=int)
        D = self.input["features"]["fragment_pair_positions"][:,3]-self.input["features"]["fragment_pair_positions"][:,2]
        ##so if you have A-B-C --> A-B-d=1, B-A-d=-1

        for j, (f1,f2,d) in enumerate(self.input["features"]["fragment_pair_relative_positions"]):
            pair_positions_to_relative_positions[:,j] = ((self.input["features"]["fragment_pair_positions"][:, [0, 1]] == [f1, f2]).all(axis=1))*(D==d)

        ##Perform matrix multiplication on the above to determine their corresponding counts
        data_feature_count["fragment_pair_relative_positions"] = csr_matrix(data_feature_count["fragment_pair_positions"]) @ csr_matrix(pair_positions_to_relative_positions).A
        data_feature_count["fragment_pairs"] = csr_matrix(data_feature_count["fragment_pair_positions"]) @ csr_matrix(pair_positions_to_pair).A

        ##Offset the feature count matrices to account for SAG
        ##I.e. a (n_data x n_feature) matrix becomes (n_data x 2*n_feature) which can be converted if wanted into (n_data x 2 x n_feature) by reshape(n_data,2,n_feature)
        n0SAG = (1-self.input["data"]["S"]).sum()
        data_feature_count_SAG = {}
        for key,val in data_feature_count.items():
            data_feature_count_SAG[key] = np.row_stack([np.column_stack([val[:n0SAG],np.zeros_like(val[:n0SAG])]),np.column_stack([np.zeros_like(val[n0SAG:]),val[n0SAG:]])])

        #Export to 'input'
        self.input["feature_count"] = data_feature_count
        self.input["feature_count_SAG"] = data_feature_count_SAG

    def export(self):
        if self.options["export_file_name"] is not None:
            with open(self.options["export_file_name"]+".pkl", 'wb') as file:
                # Step 4: Use pickle.dump to write the dictionary to the file
                pickle.dump(self.input, file)


