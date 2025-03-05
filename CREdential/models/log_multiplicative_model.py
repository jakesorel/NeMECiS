from sklearn.linear_model import Ridge
import numpy as np

class LogMultiplicativeModel:
    """
    models:

    log(Y) = intercept + features@coefs

    This corresponds to:

    Expression = (1/zeta) * product_{features} F_{reg_feature}

    where F_reg_feature = exp(coef_feature)

    and zeta = exp(-intercept)

    This can be fit with the built-in Sklearn Ridge Regression

    """
    def __init__(self,input=None,options=None):
        assert input is not None
        assert options is not None
        self.input = input
        self.options = options
        self.out = {}

    def fit(self):
        mdl = Ridge(alpha=self.options["hyperparameters"]["alpha"], fit_intercept=self.options["fit_intercept"])
        mdl.fit(self.input["Z"], self.input["Y"],
                sample_weight=self.input["weight"])
        self.out["feature_coef"] = mdl.coef_
        self.out["model_coef"] = np.array([mdl.intercept_])
        self.out["merge_coef"] = np.concatenate([self.out["feature_coef"],self.out["model_coef"]])

    def predict(self,Z):
        return Z@self.out["feature_coef"] + self.out["model_coef"]

