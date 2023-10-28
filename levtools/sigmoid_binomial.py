import numpy as np
import math
import scipy
import scipy.optimize
import scipy.stats
import pymc as pm

import arviz as az
from matplotlib import pylab as plt
import matplotlib.collections
import matplotlib.patches
from warnings import simplefilter
from abc import ABC, abstractmethod


from . import tools as t

simplefilter(action='ignore', category=DeprecationWarning)


EPSILON = np.finfo(float).eps
MAX_FLOAT = np.finfo(np.float64).max


def beta_distrib_par_estim(success_count, experiments_count, smoothing=0):
    """
    Estimating parameters of beta distribution which fits to k and n.
    This may be seen as an inversion to Bin distribution so that instead of
    k is a random variable in Bin( k | n, theta ),
    theta is the random variable such that Beta( theta | alpha, beta )

    """
    k = success_count
    n = experiments_count
    smoothing = math.log(math.exp(1) + smoothing)
    alpha = k / smoothing + 1
    beta = (n - k) / smoothing + 1
    return [alpha, beta]


def var_of_beta_distribution(alpha, beta):
    num = alpha * beta
    den = np.power(alpha + beta, 2) * (alpha + beta + 1)
    return num / den


def sigmoid(x, slope, x_offset, y_max, y_min=0, limited=True):
    exp_arg = slope * (x - x_offset)

    #if limited:
    #    exp_arg = np.clip(exp_arg, -np.log(MAX_FLOAT), np.log(MAX_FLOAT))

    range = y_max - y_min
    return range / (1 + np.exp(-exp_arg)) + y_min


def get_sigmoid_fn(slope, x_offset, y_max, limited=True):
    """
    Returns a sigmoid function parameterized by theta
    """

    return lambda x: sigmoid(x, slope, x_offset, y_max, limited=limited)


class FuncBetaPredictor(ABC):

    def __init__(self, theta=None, bounds=None, methods=[ "trust-constr", "Nelder-Mead"],
                 debug=False, norm=t.Lp_norm(p=2)):
        """
        Suggested methods: "BFGS", "trust-constr" ( a bit better but 10x slower), "Nelder-Mead" ( a bit worse )
        """
        self.methods = methods
        self.disp = debug
        self.theta = theta
        self.bounds = bounds

        # placeholders
        self.opt_desc = None
        self.x = None
        self.y = None
        self.norm = norm

        # abstract
        self.pred_func = None  # placeholder for a function. It will dynamically added in the inherited classes

    @abstractmethod
    def _get_initial_theta(self):
        pass

    def _add_data(self, x, y):
        x = t.get_np_vector(x)
        y = t.get_np_vector(y)
        inx = np.argsort(x)
        self.x = x[inx]
        self.y = y[inx]

    def _fit(self):
        thetas = []
        losses = []
        for method in self.methods:
            theta, loss = self._search_params(method)
            thetas.append(theta)
            losses.append(loss)

        best_method_inx = np.argmin(losses)
        self.theta = thetas[best_method_inx]
        self.best_method = self.methods[best_method_inx]
        return self

    def fit(self, x, y):
        self._add_data(x, y)
        return self._fit()

    def predict(self, x):
        x = t.get_np_vector(x)
        return self.pred_func(*self.theta)(x)

    def _search_params(self, method):
        theta0 = self._get_initial_theta()
        self.opt_desc = scipy.optimize.minimize(self.loss, x0=theta0,
                                                method=method, bounds=self.bounds, tol=1e-10,
                                                options={'disp': True})
        if self.disp:
            print(self.opt_desc)

        return self.opt_desc['x'], self.opt_desc['fun']

    def loss(self, theta):
        y_pred = self.pred_func(*theta)(self.x)
        err = np.abs(self.y - y_pred)
        return self.norm(err)

    def plot(self, xlabel=None, ylabel=None, title=None):
        self._plot_data()
        self._plot_pred()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()

    def _plot_pred(self):
        pred = self.predict(self.x)
        plt.plot(self.x, pred, '-', label='Prediction by ' + self.__class__.__name__)

    def _plot_data(self):
        plt.plot(self.x, self.y, 'r.', label='data')


def make_boxes(ax, xdata, ydata, xerror, yerror, facecolor='r', edgecolor='None', alpha=0.5):
    for x, y, xe, ye in zip(xdata, ydata, xerror, yerror):
        ax.add_patch(matplotlib.patches.Rectangle((x - xe / 2, y - ye / 2), xe, ye,
                                                  linewidth=1, edgecolor='r', facecolor='none'))


def plot_ratios(x, successes, experiments, label=None, axis=None):
    x = t.get_np_vector(x)
    successes = t.get_np_vector(successes)
    experiments = t.get_np_vector(experiments)

    ratios = successes / experiments

    plt.plot(x, ratios, '+', label=label)

    xdata = x
    xerror = np.ones_like(x) * 0.1

    ydata = [success / experiment for success, experiment in zip(successes, experiments)]
    yerror = [np.sqrt(var_of_beta_distribution(*beta_distrib_par_estim(success, experiment)))
              for success, experiment in zip(successes, experiments)]

    if axis is None:
        axis = plt.gca()

    make_boxes(axis, xdata, ydata, xerror, yerror)


class SigmoidPredictor(FuncBetaPredictor):

    def __init__(self,
                 *args,
                 bounds = ((-np.inf, 0),  # slope
                            (0, np.inf),  # x_offset
                            (0, 1)),  # y_max_bound
                            **kwargs):

        super(SigmoidPredictor, self).__init__(*args, bounds=bounds, **kwargs)
        self.pred_func = get_sigmoid_fn

    def _get_initial_theta(self):
        y_max_0 = self.y.max()
        slope_0, y_offset_0 = np.polyfit(self.x, self.y, 1)
        x_offset_0 = 0 if slope_0 == 0 else - y_offset_0 / slope_0
        return [slope_0, x_offset_0, y_max_0]


class SigmoidBinomialPredictor(SigmoidPredictor):

    def __init__(self, *args, **kwargs):
        super(SigmoidBinomialPredictor, self).__init__(*args, **kwargs)
        """
        Suggested methods: "L-BFGS-B", "trust-constr" ( a bit better but 10x slower), "Nelder-Mead" ( a bit worse )
        """
        self._alphas = None  # placeholder
        self._betas = None   # placeholder

    def fit(self, x, successes, experiments):
        self._add_data(x, successes, experiments)
        self._alphas, self._betas = t.zip_map(beta_distrib_par_estim, self.successes, self.experiments)
        self.loss = self._get_neg_log_likelihood
        return self._fit()

    def _add_data(self, x, successes, experiments):
        x = t.get_np_vector(x)
        inx = np.argsort(x)
        self.x = x[inx]
        self.successes = t.get_np_vector(successes)[inx]
        self.experiments = t.get_np_vector(experiments)[inx]
        self.y = self.successes / self.experiments

    def _get_neg_log_likelihood(self, theta):
        ratios_pred = get_sigmoid_fn(*theta)(self.x)
        lik_of_theta_given_data = scipy.stats.beta.pdf(ratios_pred, self._alphas, self._betas) + EPSILON
        nll = -np.sum(np.log(lik_of_theta_given_data))
        return nll

    def _plot_data(self):
        super(SigmoidBinomialPredictor, self)._plot_data()
        plot_ratios(self.x, self.successes, self.experiments, label='Binomial samples from Poisson crowd')


class SigmoidBinomialBayesianPredictor(SigmoidPredictor):

    def __init__(self, steps=3000, burnin=None, *args, **kwargs):
        """
        Suggested methods: "L-BFGS-B", "trust-constr" ( a bit better but 10x slower), "Nelder-Mead" ( a bit worse )
        """

        self.sb = SigmoidBinomialPredictor(*args, **kwargs)
        self.steps = steps
        self.burnin = steps // 2 if burnin is None else burnin
        #placeholder
        self.trace = None

    def fit(self, x, successes, experiments):
        theta0 = self.sb.fit(x, successes, experiments).theta
        [slope0, offset0, y_max0] = theta0

        model = pm.Model()
        with model:
            slope = pm.HalfNormal("slope", -slope0)
            x_offset = pm.Normal("x_offset", offset0, 40)
            y_max = pm.HalfNormal("y_max", y_max0)

            theta = [-slope, x_offset, y_max]
            crs_at_x = get_sigmoid_fn(*theta, limited=False)(self.sb.x)
            y_obs = pm.Binomial('y_obs', p=crs_at_x, n=self.sb.experiments, observed=self.sb.successes)

            # unpooled_trace = pm.sample() #2000, tune=1000, target_accept=0.9)
            #self.trace = pm.sample(self.steps, init="adapt_diag", return_inferencedata=False)

            self.mean_field = pm.fit(method='advi', callbacks=[pm.callbacks.CheckParametersConvergence()])
            self.trace = pm.sample(self.steps, tune=self.burnin, target_accept=.99)

        return self

    def predict(self, x, std=True):
        # parameter means
        y_preds = []
        for slope, x_offset, y_max in zip(self.trace['posterior']['slope'].values.reshape(-1).tolist(),
                                          self.trace['posterior']['x_offset'].values.reshape(-1).tolist(),
                                          self.trace['posterior']['y_max'].values.reshape(-1).tolist()):

            theta = [-slope, x_offset, y_max]
            y_preds.append(get_sigmoid_fn(*theta, limited=False)(x))

        y_preds = np.array(y_preds)

        pred_mean = y_preds.mean(axis=0)
        pred_std = y_preds.std(axis=0)

        if std:
            return pred_mean, pred_std
        else:
            return pred_mean

    def plot(self):
        self.sb.plot()
        pred_mean, pred_std = self.predict(self.sb.x)
        plt.plot(self.sb.x, pred_mean, '--', label='Prediction Mean by SigmoidBinomialBayesianPredictor ')
        plt.fill_between(self.sb.x, pred_mean - 3 * pred_std, pred_mean + 3 * pred_std, alpha=0.33,
                         label='Prediction Uncertainty by SigmoidBinomialBayesianPredictor ($\mu\pm3\sigma$)')

    def plot_stats(self):
        az.plot_posterior(self.trace)

        plt.figure()
        plt.title("Neg Log Likelihood of first chain")
        pm_data = az.from_pymc3(self.trace)
        plt.plot(pm_data.sample_stats.lp.values[0, :])

        for varname in self.trace.varnames:
            plt.figure()
            plt.plot(self.trace[varname])
