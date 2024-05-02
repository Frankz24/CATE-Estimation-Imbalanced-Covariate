# adapted from https://github.com/som-shahlab/RATE-experiments/blob/main/experiments/section_4_comparing_weighting_functions/qini_vs_autoc.py

import os
from typing import Callable, Iterable, Mapping, Optional, Tuple, List

import econml
from econml import grf
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
from sklearn.model_selection import KFold
from tqdm import tqdm

def rmse(predictions, targets):
    return np.sqrt(np.mean(np.square(predictions - targets)))

def aipw_func(
    X: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
    e: Optional[float] = 0.5,
    m: Optional[Callable] = None,
    params = {},
    n_folds: int = 2,
) -> np.ndarray:
    """Estimates Augmented Inverse Propensity Weight (AIPW) scores

    Args:
        X: A [num_samples, num_features] numpy array with shape representing
            N samples and p input covariates
        Y: A [num_samples, 1] numpy array representing outcomes corresponding
            to N individuals given by X.
        W: A [num_samples, 1] numpy array with W[i] = 1 if the ith individual
            was treated and 0 otherwise. Note that only binary treatments are
            currently supported.
        e: A float that represents the probability the ith subject would be
            treated (i.e., prob that W[i] = 1). We assume known propensity
            scores in this case.
        m: A function that takes in an [num_samples, num_features + 1] matrix
            representing the covariates for each subject and a final column
            representing treatment assignment and returns the estimated
            marginal outcome. If this is defined, it is used instead of
            training new nuisance parameter models/marginal outcome estimators
            from scratch. Defining m to be the ground truth marginal response
            curves can be used to calculate Oracle scores.

    Returns:
        AIPW_Scores: A [N, 1] numpy array representing the estimated
            AIPW scores for each individual
    """
    n = X.shape[0]
    e_hat = np.repeat(e, n)  # TODO: We do not yet support estimation of \hat{e}

    # Shape handling to make sure everything works with eg sklearn, econml
    if len(W.shape) == 1:
        W = W.reshape(-1, 1)
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    Y = Y.flatten()

    # AIPW scores for continuous and binary outcomes
    if m is not None:  # If m is known then we generate oracle scores
        mu_hat_1 = m(X, np.ones(n))
        mu_hat_0 = m(X, np.zeros(n))
    else:
        mu_hat_1 = np.ones((n, 1)) * -np.infty
        mu_hat_0 = np.ones((n, 1)) * -np.infty
        kf = sklearn.model_selection.KFold(n_splits=n_folds, shuffle=True)
        for train_idx, test_idx in kf.split(X):
            n_test = len(test_idx)

            # Use an honest forest to estimate baseline model/outcomes
            print(params)
            outcome_model = econml.grf.RegressionForest(**params)
            outcome_model.fit(np.hstack([X[train_idx], W[train_idx]]), Y[train_idx])

            # Predict outcomes under treatment for the held out individuals
            mu_hat_1[test_idx] = outcome_model.predict(
                np.hstack([X[test_idx], np.ones((n_test, 1))])
            )

            # Predict outcomes under control for the held out individuals
            mu_hat_0[test_idx] = outcome_model.predict(
                np.hstack([X[test_idx], np.zeros((n_test, 1))])
            )

        # Make sure we gave an estimate for every subject
        assert not np.any(mu_hat_1 == -np.infty)
        assert not np.any(mu_hat_0 == -np.infty) 

    if np.max(X[:,0])==1:
        idx_min_treated = np.intersect1d(np.where(X[:,0] == 1)[0],np.where(W == 1)[0])
        idx_maj_treated = np.intersect1d(np.where(X[:,0] == 0)[0],np.where(W == 1)[0])
        idx_min_control = np.intersect1d(np.where(X[:,0] == 1)[0],np.where(W == 0)[0])
        idx_maj_control = np.intersect1d(np.where(X[:,0] == 0)[0],np.where(W == 0)[0])
    else:
        idx_min_treated = np.intersect1d(np.where(X[:,0] >= 40)[0],np.where(W == 1)[0])
        idx_maj_treated = np.intersect1d(np.where(X[:,0] < 40)[0],np.where(W == 1)[0])
        idx_min_control = np.intersect1d(np.where(X[:,0] >= 40)[0],np.where(W == 0)[0])
        idx_maj_control = np.intersect1d(np.where(X[:,0] < 40)[0],np.where(W == 0)[0])
    print(rmse(mu_hat_1[idx_min_treated],Y[idx_min_treated]), rmse(mu_hat_1[idx_maj_treated],Y[idx_maj_treated]), rmse(mu_hat_1[idx_min_control],Y[idx_min_control]), rmse(mu_hat_1[idx_maj_control],Y[idx_maj_control]))
    # Use standard formula to estimate AIPW scores for each subject
    AIPW_scores = (
        mu_hat_1.flatten()
        - mu_hat_0.flatten()
        + W.flatten() / e_hat.flatten() * (Y - mu_hat_1.flatten())
        - (1 - W.flatten()) / (1 - e_hat.flatten()) * (Y - mu_hat_0.flatten())
    )

    return AIPW_scores


def ipw_func(
    X: np.ndarray, Y: np.ndarray, W: np.ndarray, e: Optional[float] = 0.5
) -> np.ndarray:
    """Estimates Inverse Propensity Weight (IPW) scores

    Args:
        X: A [num_samples, num_features] numpy array with shape representing
            N samples each with d-dimensional input covariates
        Y: A [num_samples, 1] numpy array representing outcomes corresponding
            to N individuals with covariates given by X.
        W: A [num_samples, 1] numpy array with W[i] = 1 if the ith individual
            was treated and 0 otherwise. Note that only binary treatments are
            currently supported.
        e: A float that represents the probability the ith subject would be
            treated (i.e., prob that W[i] = 1). We assume known propensity
            scores in this case.

    Returns:
        IPW_Scores: A [N, 1] numpy array representing the estimated AIPW scores
            for each individual
    """
    e_hat = np.repeat(e, X.shape[0])
    W = W.flatten()
    Y = Y.flatten()
    e_hat = e_hat.flatten()
    IPW_scores = (W * Y / e_hat) - ((1.0 - W) * Y / (1.0 - e_hat))
    return IPW_scores


def get_scores(X, Y, W, e=0.5, m=None, params={"n_estimator":1000}, scoring_type="AIPW") -> np.ndarray:
    """Estimate scores (proxies for the CATE) for each subject

    Args:
        X: A [num_samples, num_features] numpy array with shape representing
            N samples each with d-dimensional input covariates
        Y: A [num_samples, 1] numpy array representing outcomes corresponding
            to N individuals with covariates given by X.
        W: A [num_samples, 1] numpy array with W[i] = 1 if the ith individual
            was treated and 0 otherwise. Note that only binary treatments are
            currently supported.
        e: A float that represents the probability the ith subject would be
            treated (i.e., prob that W[i] = 1)
        m: A function that takes in an [num_samples, num_features + 1] matrix
            representing the covariates for each subject and a final column
            representing treatment assignment and returns the estimated
            marginal outcome. If this is defined, it is used instead of
            training new nuisance parameter models/marginal outcome estimators
            from scratch. Defining m to be the ground truth marginal response
            curves can be used to calculate Oracle scores.
        scoring_type: One of 'IPW', 'AIPW', or 'Oracle', representing inverse
            propensity weighted scores, augmented inverse propensity weighted
            scores, or 'Oracle' scores which are AIPW scores where the true
            conditional response surfaces are known and given. Note that
            the 'Oracle' scores are not the exact individualized treamtent
            difference, but rather the true expected difference in potential
            outcomes conditioned on covariates.

    Returns:
        A [num_samples, ] numpy array where the ith element of the array
        represents either an IPW, AIPW, or Oracle score for that subject.
        These scores are nearly unbiased proxies for the CATE.
    """
    if scoring_type == "IPW":
        return ipw_func(X, Y, W, e)
    elif scoring_type == "AIPW":
        return aipw_func(X, Y, W, e, params=params)
    elif scoring_type == "Oracle" and m is not None:
        return aipw_func(X, Y, W, e, m)
    elif scoring_type == "Oracle" and m is None:
        raise ValueError(
            "Cannot provide oracle scores without ground "
            "truth marginal response function!"
        )
    else:
        raise ValueError(f"Scoring method {scoring_type} not supported")
    

def scaled_RATE(sorted_scores: np.ndarray, method: str = "AUTOC") -> float:
    """Calculate a centered, scaled estimate of the RATE

    Calculates a centered and scaled version of the Qini (and AUTOC).
    Given some fixed sample size, n, the variance of the weights
    applied to each doubly robust score in the AUTOC is constrained
    to be 1 while the variance of the weights applied to these same
    scores for the Qini coefficient tends toward 0.5 for large n. In this
    function, we rescale the Qini coefficient weights to also have variance
    1 in order to fairly compare the two metrics. While this rescaling
    changes the value of the point estimate for these methods, it does
    not change the statistical power of the two approaches.

    Args:
        sorted_scores: A [num_samples, ] array containing "scores" (nearly
            unbiased proxies of the CATE, such as IPW or AIPW scores), sorted
            in order of the priority scores output by the prioritization rule
            under consideration.
        method: A string indicating which of the two methods ("AUTOC" vs.
            "QINI") should be calculated

    Returns:
        RATE (float): A point estimate of the Rank-Weighted Average Treatment
            Effect, the exact form of which is specified by `method`.
    """
    n = len(sorted_scores)
    H: np.ndarray = np.array([])
    if method == "AUTOC":
        rev_inv_rank = 1.0 / np.arange(n, 0, -1)
        # AUTOC should be centered/scaled by default
        H = np.flip(np.cumsum(rev_inv_rank)) - 1
    elif method == "QINI":
        rev_sort_u = np.arange(n, 0, -1) / n
        # We explicitly center and scale the Qini coefficient
        H = (rev_sort_u - np.mean(rev_sort_u)) / np.std(rev_sort_u)
    else:
        raise ValueError(f"Method {method} is not supported")
    RATE = np.mean(H * sorted_scores)
    return RATE
