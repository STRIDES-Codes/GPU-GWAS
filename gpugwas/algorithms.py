"""Module for GWAS analysis algorithms."""
## For usage/API see:
## Logistic regrssion: https://gist.github.com/VibhuJawa/316792903e90b7379c06a9c9cb4187ef
## Both: https://gist.github.com/VibhuJawa/b33927bda957355ebdc9bc261a24b142

from cuml import linear_model as cuml_linear_model
import cupy as cp
import cudf
import numpy as np
import scipy.stats as stat

from cuml import PCA
import cupy as cp
from cuml import linear_model as cuml_linear_model
import numpy as np
from scipy import stats
import cudf

def PCA_concat(df,components=100):
    pca_float = PCA(n_components = 2)
    pca_float.fit(df[df.columns[df.dtypes == np.float32]])
    scores = pca_float.transform(df[df.columns[df.dtypes == np.float32]])
    return cudf.concat([df, scores],axis=1)


## Port of sklearn Logistic Regression
# From: https://gist.github.com/rspeare/77061e6e317896be29c6de9a85db301d
class cuml_LogisticReg:
    """
    Wrapper Class for Logistic Regression which has the usual sklearn instance
    in an attribute self.model, and pvalues, z scores and estimated
    errors for each coefficient in

    self.z_scores
    self.p_values
    self.coefficients

    as well as the negative hessian of the log Likelihood (Fisher information)

    self.F_ij
    """

    def __init__(self, *args, **kwargs):  # ,**kwargs):
        self.model = cuml_linear_model.LogisticRegression(*args, **kwargs)  # ,**args)

    def fit(self, X, y):
        self.model.fit(X, y)
        #### Get p-values for the fitted model ####
        denom = 2.0 * (1.0 + np.cosh(self.model.decision_function(X)))
        denom = cp.tile(denom, (X.shape[1], 1)).T
        F_ij = cp.dot((X / denom).T, X)  ## Fisher Information Matrix
        Cramer_Rao = cp.linalg.inv(F_ij)  ## Inverse Information Matrix
        sigma_estimates = cp.sqrt(cp.diagonal(Cramer_Rao))

        ## Changed below to make it equal to sklearn
        z_scores = (
            self.model.coef_.flatten() / sigma_estimates
        )  # z-score for eaach model coefficient
        # z_scores = self.model.coef_[0]/sigma_estimates # z-score for eaach model coefficient

        # serial on cpu but only n_features so should not be too bad
        ## cna look into gpu accerattion if needed
        p_values = [
            stat.norm.sf(abs(x.item())) * 2 for x in z_scores
        ]  ### two tailed test for p-values

        ### In case we need confidence intervals
        # from: https://gist.github.com/rspeare/77061e6e317896be29c6de9a85db301d#gistcomment-2267786
        # alpha = 0.05
        # q = stats.norm.ppf(1 - alpha / 2)
        # lower = self.model.coef_[0] - q * sigma_estimates
        # upper = self.model.coef_[0] + q * sigma_estimates
        # self.conf_int = np.dstack((lower, upper))[0]

        self.z_scores = z_scores
        self.p_values = p_values
        self.sigma_estimates = sigma_estimates
        self.F_ij = F_ij


# PORT OF : https://stackoverflow.com/a/42677750/7904797
class cuml_LinearReg:
    """
    Wrapper Class for Linear Regression which has the usual sklearn instance
    in an attribute self.model, and pvalues, z scores and estimated
    errors for each coefficient

    self.t_values
    self.p_values
    self.standard_errors
    """

    def __init__(self, *args, **kwargs):  # ,**kwargs):
        self.model = cuml_linear_model.LinearRegression(*args, **kwargs)  # ,**args)

    def fit(self, X, y):
        self.model.fit(X, y)
        predictions = self.model.predict(X)

        ### append ones to the features
        newX = cp.ones(shape=(X.shape[0], X.shape[1] + 1), dtype=X.dtype)
        newX[:, 1:] = X

        len_delta = len(newX) - len(newX[0])

        MSE = ((y - predictions) ** 2).sum() / (len_delta)
        del predictions

        ### append intercept_ to end of coef_
        params = cp.zeros(
            shape=(len(self.model.coef_) + 1), dtype=self.model.coef_.dtype
        )
        params[0] = self.model.intercept_
        params[1:] = self.model.coef_

        ## TODO: Below might be memory intensive, look into this if needed
        var_b = MSE * (cp.linalg.inv(cp.dot(newX.T, newX)).diagonal())

        ### Free up memory
        del newX

        sd_b = cp.sqrt(var_b)
        ts_b = params / sd_b

        ### TODO: May be can do on gpu if slow (should not be as ts_b is just n_features long
        p_values = [2 * (1 - stats.t.cdf(np.abs(i.get()), len_delta)) for i in ts_b]

        sd_b = np.round(sd_b, 3)
        ts_b = np.round(ts_b, 3)
        p_values = np.round(p_values, 3)
        params = np.round(params, 4)

        self.coefficients = params
        self.standard_errors = sd_b
        self.t_values = ts_b
        self.p_values = p_values
