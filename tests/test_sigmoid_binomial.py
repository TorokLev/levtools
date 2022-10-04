import numpy as np

from levtools import sigmoid_binomial as sb


def generate_sigmoid_binomial_samples(xs, theta, poisson_means):

    ratios_gt = sb.get_sigmoid_fn(*theta)(xs)

    experiments = np.random.poisson(lam=poisson_means, size=len(xs))
    successes = np.random.binomial(n=experiments, p=ratios_gt, size=len(xs))

    return ratios_gt, successes, experiments


def test_fitting_sigmoid_predictor():

    x = np.linspace(0, 10)
    y = sb.sigmoid(x, slope=-1, x_offset=5, y_max=1)

    clf = sb.SigmoidPredictor()
    clf.fit(x, y)
    ys_predicted = clf.predict(x)
    np.testing.assert_array_almost_equal(ys_predicted, y, decimal=3)


def test_sigmoid_beta_predictor_fitting():
    prices = np.linspace(20, 30, 7)
    population = np.round(300 / (prices - 15))
    theta_gt = [-0.2, 24, 1.0]
    cr_gt, successes, experiments = generate_sigmoid_binomial_samples(prices, theta_gt, poisson_means=population)

    clf = sb.SigmoidBinomialPredictor(methods=['trust-constr'])
    clf.fit(prices, successes, experiments)
    cr_predicted = clf.predict(prices)

    assert len(cr_predicted) == len(successes)


def test_sigmoid_beta_bayesian_predictor_fitting():
    prices = np.linspace(20, 30, 7)
    population = np.round(300 / (prices - 15))
    theta_gt = [-0.2, 24, 1.0]
    cr_gt, successes, experiments = generate_sigmoid_binomial_samples(prices, theta_gt, poisson_means=population)

    clf = sb.SigmoidBinomialBayesianPredictor(steps=150)
    clf.fit(prices, successes, experiments)
    cr_predicted, cr_pred_std = clf.predict(prices)

    assert len(cr_predicted) == len(successes)
    assert len(cr_pred_std) == len(successes)
