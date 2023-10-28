import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from matplotlib import pylab as plt


def isiterable(x):
    return hasattr(x, '__iter__')


def zip_map(map_fn, *inputs):
    processed = [map_fn(*params) for params in zip(*inputs)]
    if isiterable(processed[0]):
        return zip(*processed)
    else:
        return processed


def np_clip(arr, min, max):

    def clipper(val, min, max):
        if val < min:
            return min
        if val > max:
            return max
        return val

    return np.array([clipper(value, min, max) for value in arr])


def Lp_norm(p, w=None):
    if w is None:
        w = np.ones_like(p)
    return lambda x: np.power(np.sum(w * np.power(x, p)), 1/p)


def get_evenly_balanced_bins(data, n_slices, weights=None):
    if weights is not None:
        data = [d for d, w in zip(data, weights) for _ in range(w)]

    percent_slices = np.linspace(0, 100, n_slices + 1)
    bin_bounds = np.percentile(data, percent_slices)
    return list(zip(bin_bounds[:-1], bin_bounds[1:]))


def find_largest_n_for_evenly_balanced_splits(data, max_slices=20, weights=None):
    """
    max_slices if None it is trying to find the largest resolution in which no bin is empty (upto 20)
    """

    for max_slices in np.arange(max_slices, 2, -1):
        bin_bounds = get_evenly_balanced_bins(data, weights=weights, n_slices=max_slices)

        lefts = list(zip(*bin_bounds))[0]
        rights = list(zip(*bin_bounds))[1]

        if (np.diff(lefts) == 0).sum() == 0 and (np.diff(rights) == 0).sum() == 0:
            break

    return bin_bounds


def quantize_to_bins(input_col, bins, to_return='index'):
    """
    Can return index, median, middle, mean or count (size)
    """

    output_col = get_np_vector(input_col).copy()

    for inx, (low, high) in enumerate(bins):

        if inx == 0:
            lower_limit = (output_col >= low)
        else:
            lower_limit = (output_col > low)

        mask = lower_limit & (output_col <= high)

        input_values = input_col[mask]

        if to_return == 'middle':
            output_col[mask] = (low + high) / 2
        elif to_return == 'index':
            output_col[mask] = inx
        elif to_return == 'median':
            output_col[mask] = np.median(input_values)
        elif to_return == 'mean':
            output_col[mask] = np.mean(input_values)
        elif to_return == 'count' or to_return == 'size':
            output_col[mask] = len(input_values)

    return output_col


def _check_matrix_ndarray(x):
    if len(x.shape) == 2:
        return x
    elif len(x.shape) == 1:
        return x.reshape(1, -1) # row vector
    else:
        raise Exception("Numpy array with neither 2 or 1 dimensions given")


def get_np_matrix(x):
    if isinstance(x, pd.DataFrame):
        return x.values
    elif isinstance(x, pd.Series):
        return x.values.reshape(-1,1) # column vector
    elif isinstance(x, np.ndarray):
        return _check_matrix_ndarray(x)
    elif isinstance(x, list):
        return _check_matrix_ndarray(np.array(x))
    else:
        raise Exception("Variable cannot be represented as matrix")


def _check_vector_ndarray(x):
    if len(x.shape) == 1:
        return x
    elif x.shape[1] == 1:
        return x[:, 0]
    else:
        raise Exception("Variable cannot be represented as vector")


def get_np_vector(x):
    if isinstance(x, pd.DataFrame):
        return _check_vector_ndarray(x.values)
    elif isinstance(x, pd.Series):
        return x.values
    elif isinstance(x, np.ndarray):
        return _check_vector_ndarray(x)
    elif isinstance(x, list):
        x = np.array(x)
        return _check_vector_ndarray(x)
    else:
        raise Exception("Variable cannot be represented as vector")


def sample_uniform_date_range(start, end, size=1):
    start = pd.Timestamp(start).timestamp()
    end = pd.Timestamp(end).timestamp()

    times = pd.to_datetime(np.random.uniform(start, end, size=np.prod(size)) * 1e+9).tolist()

    if np.isscalar(size):
        if size == 1:
            return times[0]
        else:
            return np.reshape(times, size)[0]

    return np.reshape(times, size)


def flatten(list2d):
    return list(itertools.chain.from_iterable(list2d))


EPSILON = np.finfo(float).eps
NEG = (-np.inf, -EPSILON)
POS = (EPSILON, np.inf)
NEG0 = (-np.inf, 0)
POS0 = (0, np.inf)
ZERO_ONE = (0, 1.0)
ANY = (-np.inf, np.inf)


def generate_sample_in_bounds(bounds):
    if bounds == POS or bounds == POS0:
        return np.abs(np.random.randn())
    if bounds == NEG or bounds == NEG0:
        return -np.abs(np.random.randn())
    if bounds == ANY:
        return np.random.randn()
    return np.random.uniform(bounds[0], bounds[1])


class VectorPacker:
    def __init__(self, sizes_dict, bounds_dict=None):
        self.sizes_dict = sizes_dict
        self.len = np.sum([np.prod(size) for size in sizes_dict.values()])
        self.bounds_dict = bounds_dict
        if bounds_dict:
            self.bounds_vect = flatten([[bounds_dict[key]] * np.prod(sizes_dict[key])
                                        for key, size in self.sizes_dict.items()])
        else:
            self.bounds_vect = None

    def generate_random(self):
        return [generate_sample_in_bounds(bounds) for bounds in self.bounds_vect]

    def to_dict(self, input_vector):
        start = 0
        unpacked = {}
        for key, size in self.sizes_dict.items():
            vector = input_vector[start:start + np.prod(size)]
            if size == 1:
                unpacked[key] = vector[0]
            else:
                unpacked[key] = np.reshape(vector, size)
            start += np.prod(size)
        return unpacked

    def to_vector(self, packed):
        start = 0
        output_vector = np.zeros(self.len)
        for key, size in self.sizes_dict.items():
            vector = np.reshape(packed[key], (-1))
            output_vector[start:start + np.prod(size)] = vector
            start += len(vector)
        return output_vector


def duplicate_pd(df, n):
    return pd.concat([df] * n)


def one_encoder(x):
    """
    from a one dimensional array of integers generates a matrix of 0's and 1's

    Each row will have exactly one 1 and the rest is full of zeros.
    The index of the 1 in the row coresponds to the integer value in the input vector.
    Example:
        one_encoder(np.array([0,2,1]))
    Returns:
        np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

    """

    n = len(x)
    m = x.max() - x.min() + 1

    y = np.zeros((n, m))
    y[np.arange(n), x - x.min()] = 1
    return y


class OneHotEncoder():
    def __init__(self, prefix = "", mappings=None):
        self.mappings = mappings
        self.prefix = prefix

    def fit(self, X):
        pass

    def transform(self, X):
        pass


def get_distribution_mode(x, max_slices):
    bins = find_largest_n_for_evenly_balanced_splits(x, max_slices=max_slices)
    bins = np.array(bins)
    v = (bins[:,1] - bins[:, 0])
    return bins[v.argmin(), :].mean()


import itertools


class Poisson:
    def __init__(self, mu=None):
        if mu:
            self.distr = scipy.stats.poisson(mu)
        else:
            self.distr = scipy.stats.poisson

    def pdf(self, data):
        return self.distr.pmf(data)

    def cdf(self, data):
        return self.distr.pmf(data).cumsum()

    @staticmethod
    def fit(data):
        return np.array(data).mean()


class FindDistribution:
    distrib_descriptors = [{'class': scipy.stats.norm, 'name': "Normal", 'param_bounds': ['any', 'pos']},
                           {'class': Poisson, 'name': "Poisson", 'param_bounds': ['pos']},
                           {'class': scipy.stats.laplace, 'name': "Laplace", 'param_bounds': ['any', 'pos']},
                           {'class': scipy.stats.logistic, 'name': "Logistic", 'param_bounds': ['any', 'pos']},
                           {'class': scipy.stats.rayleigh, 'name': "Rayleigh", 'param_bounds': ['any', 'pos']},
                           {'class': scipy.stats.cauchy, 'name': "Cauchy", 'param_bounds': ['any', 'pos']},
                           {'class': scipy.stats.expon, 'name': "Expon", 'param_bounds': ['any', 'pos']},
                           {'class': scipy.stats.exponnorm, 'name': "ExponNorm", 'param_bounds': ['pos', 'any', 'pos']},
                           {'class': scipy.stats.gamma, 'name': "Gamma", 'param_bounds': ['any', 'any', 'pos']},
                           {'class': scipy.stats.lognorm, 'name': "LogNorm", 'param_bounds': ['pos', 'any', 'pos']},
                           {'class': scipy.stats.loglaplace, 'name': "LogLaplace",
                            'param_bounds': ['pos', 'any', 'pos']},
                           {'class': scipy.stats.laplace_asymmetric, 'name': "Laplace Asym",
                            'param_bounds': ['pos', 'any', 'pos']}
                           ]

    def __init__(self, loss_type, bins=30, repeat_times=1, best_or_avg='best'):  # loss = neg_likelihood, neg_log_likelihood

        loss_mapper = \
            {'ks': {'loss_func': FindDistribution.get_ks_test, 'space': 'cum_hist'},
             'neg_log_lik': {'loss_func': FindDistribution.get_neg_log_lik, 'space': 'data'},
             'neg_lik': {'loss_func': FindDistribution.get_neg_lik, 'space': 'data'},
             'kldiv': {'loss_func': FindDistribution.get_kldiv, 'space': 'hist'},
             'sym_kldiv': {'loss_func': FindDistribution.get_sym_kldiv, 'space': 'hist'},
             'L2': {'loss_func': FindDistribution.get_L2, 'space': 'hist', 'bins': bins}}

        self.loss_type = loss_type
        self.loss_func = loss_mapper[loss_type]['loss_func']
        self.space = loss_mapper[loss_type]['space']
        self.bins = bins if 'bins' not in loss_mapper[loss_type] else loss_mapper[loss_type]['bins']
        self.repeat_times = repeat_times
        self.best_or_avg = best_or_avg

    @staticmethod
    def get_L2(pdf, hist):
        x = pdf / np.sum(pdf)
        y = hist
        err = x - y
        return np.sum(err * err)

    @staticmethod
    def get_neg_log_lik(dist, xs, epsilon=0.000001):
        lik = dist.pdf(xs)
        lik[lik == 0] = epsilon
        return -np.sum(np.log(lik))

    @staticmethod
    def get_neg_lik(dist, xs, epsilon=0.000001):
        lik = dist.pdf(xs)
        lik[lik == 0] = epsilon
        return -np.sum(np.log(lik))

    @staticmethod
    def get_kldiv(pdf, hist, epsilon=0.000001):
        p = pdf / np.sum(pdf)
        q = hist
        q[q == 0] = epsilon

        return np.sum(p * np.log(p / q))

    @staticmethod
    def get_sym_kldiv(pdf, hist):
        p = pdf / np.sum(pdf)
        q = hist
        return FindDistribution.get_kldiv(p, q) + FindDistribution.get_kldiv(q, p)

    @staticmethod
    def get_ks_test(cdf, cum_hist):
        """
        Kolmogorov Smirnov test
        """
        return np.max(np.abs(cdf - cum_hist))

    @staticmethod
    def limit_to_par_bounds(parameters, parameter_ranges):

        return [np.abs(parameter_item) if range == 'pos' else (
            -np.abs(parameter_item) if range == 'neg' else parameter_item)
                for parameter_item, range in zip(parameters, parameter_ranges)]

    def get_loss_value(self, parameters):
        parameters = self.limit_to_par_bounds(parameters, self.cur_param_bounds)
        distr = self.cur_distr_class(*parameters)  # TODO: Ugly

        if self.space == 'hist':
            return self.loss_func(pdf=distr.pdf(self.data_xscan), hist=self.data_hist)
        elif self.space == 'cum_hist':
            return self.loss_func(cdf=distr.cdf(self.data_xscan), cum_hist=self.data_cum_hist)
        elif self.space == 'data':
            return self.loss_func(dist=distr, xs=self.data)

        # print(f"> {par} : {loss}")

    def _calc_histogram(self, x):
        counts, scan = np.histogram(x, bins=self.bins)
        self.data = x
        self.data_xscan = (scan[:-1] + scan[1:]) / 2
        self.data_hist = counts / counts.sum()
        self.data_cum_hist = np.cumsum(self.data_hist)

    def fit_one_distrib(self, x, distrib_name, noise_level=0.01):

        distrib_desc = [distrib for distrib in FindDistribution.distrib_descriptors if distrib['name'] == distrib_name][0]
        n_pars = len(distrib_desc['param_bounds'])

        par0 = distrib_desc['class'].fit(data=x)
        par0 = par0 + np.random.randn(n_pars) * (np.array(par0) * noise_level)
        par0 = self.limit_to_par_bounds(par0, parameter_ranges=distrib_desc['param_bounds'])

        self.cur_distr_class = distrib_desc['class']  # TODO: Ugly
        self.cur_param_bounds = distrib_desc['param_bounds']

        opt = scipy.optimize.minimize(self.get_loss_value, par0)

        opt_par = self.limit_to_par_bounds(opt['x'], parameter_ranges=distrib_desc['param_bounds'])
        opt_loss = opt['fun']  # self.get_loss_value(opt_par)

        return {'name': distrib_name, 'par': opt_par, 'n_params': len(opt_par), 'loss': opt_loss,
                'class': distrib_desc['class']}

    @staticmethod
    def _get_mean_of_res(res):
        name = res.iloc[0]['name']
        n_params = res.iloc[0]['name']
        _class = res.iloc[0]['class']
        loss = res['loss'].mean()
        return pd.Series({'name': name, 'n_params': n_params, 'loss': loss, 'class': _class})

    def fit_repeated_times(self, distribution, x):

        self._calc_histogram(x)
        res = [self.fit_one_distrib(x, distribution['name'], noise_level=0.0 if ix > 0 else 0)
               for ix in range(self.repeat_times)]

        pd.set_option('use_inf_as_na', True)
        res_df = pd.DataFrame.from_records(res).dropna(axis=0).sort_values(by='loss', ascending=True)

        if self.best_or_avg == 'best':
            if len(res_df) > 0:
                return res_df.iloc[0]
            else:
                print("Cannot find non failing distribution ", distribution['name'], "parameterization: ", self.loss_type)
        elif self.best_or_avg == 'avg':
            return self._get_mean_of_res(res_df)

    def fit(self, x):

        res = pd.concat([self.fit_repeated_times(distribution, x)
                         for distribution in FindDistribution.distrib_descriptors], axis=1).T

        self.res = res.sort_values(by='loss', ascending=True)
        return self.res

    def plot(self, limit_to_params=-1, top_n=None, distribution=None):

        df = self.res.query(f'n_params == {limit_to_params}') if limit_to_params > 0 else self.res
        df = df.query(f'name == "{distribution}"') if distribution else df

        plt.bar(self.data_xscan, self.data_hist, label='data')
        for _, res_row in itertools.islice(df.iterrows(), top_n):
            distr = res_row['class'](*res_row['par'])
            distr_normalizer = np.sum(distr.pdf(self.data_xscan))
            plt.plot(self.data_xscan, distr.pdf(self.data_xscan) / distr_normalizer, label=res_row['name'])

        plt.legend()


def pdf_normalizer(frozen_distr, data, epsilon=0.000001):
    cf = frozen_distr.cdf(data)
    cf_bounded = np.clip(cf, epsilon, 1 - epsilon)
    return scipy.special.erfinv(cf_bounded * 2 - 1)

