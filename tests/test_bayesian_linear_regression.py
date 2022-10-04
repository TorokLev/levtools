import pandas as pd
import numpy as np

from levtools import bayesian_linear_regression as blr

def get_dataset():
    N = 30
    df = pd.DataFrame({"x": np.sort(np.random.rand(N))})

    eps_sigma_sq = 1.0
    offset = 1
    slope = 2
    df["y"] = offset + slope * df["x"] + np.random.randn(N) * eps_sigma_sq
    return df

def test_lin_reg_predict_proba():

    df = get_dataset()
    clf = blr.RobustLinearModel().fit(df['x'], df['y'])
    likelihoods = clf.predict_proba(df['x'].values, df['y'].values)


def test_bootstrapped_lin_reg():

    df = get_dataset()

    clf = blr.BootstrapModel(model=blr.RobustLinearModel, resample_count=1000)
    clf.fit(df['x'], df['y'])
    y_pred, y_pred_std = clf.predict(df['x'])


def test_bayesian_lin_reg():
    df = get_dataset()

    clf = blr.BayesianLinReg(steps=10)
    clf.fit(df['x'], df['y'])
    #y_pred, y_pred = clf.predict(df['x'])

