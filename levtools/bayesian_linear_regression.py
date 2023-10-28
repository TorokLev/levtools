import arviz as az
import pandas as pd
import scipy.stats
from matplotlib import pylab as plt
import numpy as np
import pymc as pm
from warnings import simplefilter

from . import tools as t

simplefilter(action='ignore', category=DeprecationWarning)


class DistanceFromLinear:
    def __init__(self, slope, y_offset):
        self.normal, self.offset = DistanceFromLinear.convert_lin_reg_params_to_normal_form(slope, y_offset)

    @staticmethod
    def convert_lin_reg_params_to_normal_form(slope, y_offset):
        normal = np.array([-slope, 1])
        length = np.linalg.norm(normal, ord=2)
        normal = normal / length
        offset = np.array([- y_offset / slope, 0])
        return normal, offset

    def get_distance(self, xs, ys):
        xs = np.array(xs).reshape(-1, 1) - self.offset[0]
        ys = np.array(ys).reshape(-1, 1) - self.offset[1]
        x = np.concatenate([xs, ys], axis=1)
        dist = np.abs(np.dot(x, self.normal.reshape(-1, 1)))
        return np.squeeze(dist, axis=1)


class LinearModel:
    parameter_names = ['slope', 'y_offset']

    def __init__(self, theta=None):
        self.theta = theta

        # placeholder
        self.err_std = None

    def fit(self, xs, ys, weights=None):
        try:
            self.theta = np.polyfit(xs, ys, 1, w=weights)
        except:
            print("Bad fit")
            return self

        return self

    def predict(self, xs):
        slope, offset = self.theta
        return slope * xs + offset


class RobustLinearModel(LinearModel):
    parameter_names = ['slope', 'y_offset']

    def __init__(self, theta=None, outlier_lik_ratio_threshold=0.05):
        self.theta = theta
        self.outlier_lik_ratio_threshold = outlier_lik_ratio_threshold

        # placeholder
        self.err_std = None

    def fit(self, xs, ys, weights=None):
        xs = t.get_np_vector(xs)
        ys = t.get_np_vector(ys)

        try:
            self.theta = np.polyfit(xs, ys, 1, w=weights)
        except:
            print("Bad fit")
            return self

        if self.outlier_lik_ratio_threshold > 0:
            err = DistanceFromLinear(slope=self.theta[0], y_offset=self.theta[1]).get_distance(xs, ys)

            self.err_std = np.std(err)
            liks = scipy.stats.norm(0, self.err_std).pdf(err)
            pdf_at_zero = scipy.stats.norm(0, self.err_std).pdf(0)
            ratio = liks / pdf_at_zero
            keep_mask = ratio >= self.outlier_lik_ratio_threshold
            xs = xs[keep_mask]
            ys = ys[keep_mask]

            try:
                self.theta = np.polyfit(xs, ys, 1, w=weights)
            except:
                print("Bad fit")
                return self

        return self

    def predict_proba(self, xs, ys):
        err = self.predict(xs) - ys
        return scipy.stats.norm(0, self.err_std).pdf(err)


class BootstrapModel(object):
    def __init__(self, model, model_params={}, resample_count=100):
        """
        model: can be LinearModel, RobustLinearModel,
        """
        self.model = model
        self.model_params = model_params
        self.resample_count = resample_count
        # pleceholders
        self.thetas = None
        self.theta_mean = None
        self.x = None
        self.y = None
        self.xrange = None

    def _bootstrap_samples(self, v, resample_size=None):
        bootstrapped_index = np.random.choice(len(v), size=len(v) * 2, replace=True)
        return np.array(v)[bootstrapped_index]

    def _bootstrap_fit_1(self, xs, ys, weights=None):
        xs_bootstrapped, ys_boostrapped = list(zip(*self._bootstrap_samples(list(zip(xs, ys)))))

        model = self.model(**self.model_params).fit(np.array(xs_bootstrapped), np.array(ys_boostrapped), weights=weights)

        return model.theta

    def fit(self, x, y, weights=None):
        x = t.get_np_vector(x)
        y = t.get_np_vector(y)
        inx = np.argsort(x)
        self.x = x[inx]
        self.y = y[inx]

        self.thetas = []
        for _ in np.arange(self.resample_count):
            theta = self._bootstrap_fit_1(x, y)
            if theta is not None:
                self.thetas.append(theta)

        pred_mean = self.predict(self.x, std=False)
        mean_model = self.model(**self.model_params).fit(self.x, pred_mean, weights=weights)
        self.theta_mean = mean_model.theta

        self.xrange = (np.min(x), np.max(x))
        return self

    def _plot_1(self, theta, color='b-', label=None, alpha=None):
        xstart, xend = self.xrange
        xscan = np.linspace(xstart, xend, 100)
        yscan = self.model(theta=theta).predict(xscan)
        plt.plot(xscan, yscan, color, label=label, alpha=alpha)

    def predict(self, xs, std=True):
        xs = t.get_np_vector(xs)
        predict_results = np.concatenate([self.model(theta=theta).predict(xs).reshape(-1, 1)
                                          for theta in self.thetas], axis=1)

        y_pred_mean = predict_results.mean(axis=1)
        if not std:
            return y_pred_mean

        y_pred_std = predict_results.std(axis=1)
        return y_pred_mean, y_pred_std

    def plot(self, predict_samples=False):

        plt.plot(self.x, self.y, 'r.', label='data')

        if predict_samples:
            for theta in self.thetas:
                self._plot_1(theta)

        pred_mean, pred_std = self.predict(self.x, std=True)
        plt.plot(self.x, pred_mean, 'g-', label='Prediction Mean')
        plt.fill_between(self.x, pred_mean - 3 * pred_std, pred_mean + 3 * pred_std, alpha=0.33,
                         label='Uncertainty Interval ($\mu\pm3\sigma$)')
        plt.legend()

    def plot_stats(self):

        thetas_df = pd.DataFrame(self.thetas, columns=self.model(**self.model_params).parameter_names)
        for parameter_name in thetas_df.columns:
            plt.figure()
            plt.hist(thetas_df[parameter_name], bins=30)
            plt.title(parameter_name)


class BayesianLinReg():
    def __init__(self, intercept_prior_std=10, slope_prior_std=10, obs_noise_prior_std=1, steps=300, burnin=100):
        self.intercept_prior_std = intercept_prior_std
        self.slope_prior_std = slope_prior_std
        self.obs_noise_prior_std = obs_noise_prior_std
        self.steps = steps
        self.burnin = burnin
        # placeholders
        self.x = None
        self.y = None

    def fit(self, x, y):
        x = t.get_np_vector(x)
        y = t.get_np_vector(y)
        inx = np.argsort(x)
        self.x = x[inx]
        self.y = y[inx]

        with pm.Model() as bm:
            # Priors
            intercept = pm.Normal('y_offset', mu=0, sigma=self.intercept_prior_std)
            slope = pm.Normal('slope', mu=0, sigma=self.slope_prior_std)
            obs_noise = pm.HalfNormal('obs_noise', sigma=self.obs_noise_prior_std)

            # Deterministics part
            mu = intercept + slope * x

            # Likelihood 
            y_likelihood = pm.Normal('Ylikelihood', mu=mu, sigma=obs_noise, observed=y)
            self.trace = pm.sample(self.steps)
            return self

    def get_posterior_samples(self, x):
        return np.array([(step['slope'] * x + step['y_offset']) for step in self.trace[self.burnin:]])

    def predict(self, x, std=True):
        """
        returns matrix steps (w/o burnout) x len(x) matrix
        """
        samples = self.get_posterior_samples(x)
        pred_mean = samples.mean(axis=0)

        if not std:
            return pred_mean

        pred_std = samples.std(axis=0)
        return pred_mean, pred_std

    def plot(self):
        plt.scatter(self.x, self.y, c='k', zorder=10, label='Data')

        pred_mean, pred_std = self.predict(self.x, std=True)
        plt.plot(self.x, pred_mean, label='Prediction Mean')
        plt.fill_between(self.x, pred_mean - 3 * pred_std, pred_mean + 3 * pred_std, alpha=0.33,
                         label='Uncertainty Interval ($\mu\pm3\sigma$)')

    def plot_stats(self):
        az.plot_posterior(self.trace)
