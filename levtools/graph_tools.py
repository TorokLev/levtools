import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections
import matplotlib.patches

import sigmoid_binomial as sb


def make_boxes_boxes(ax, xdata, ydata, xerror, yerror, facecolor='r', edgecolor='None', alpha=0.5):
    for x, y, xe, ye in zip(xdata, ydata, xerror, yerror):
        ax.add_patch(matplotlib.patches.Rectangle((x - xe / 2, y - ye / 2), xe, ye,
                                                  linewidth=1, edgecolor='r', facecolor='none'))


def plot_ratios(xs, successes, experiments, label=None, axis=None):
    ratios = successes / experiments

    plt.plot(xs, ratios, '+', label=label)

    xdata = xs
    xerror = np.ones_like(xs) * 0.1

    ydata = [success / experiment for success, experiment in zip(successes, experiments)]
    yerror = [np.sqrt(sb.var_of_beta_distribution(*sb.beta_distrib_par_estim(success, experiment)))
              for success, experiment in zip(successes, experiments)]

    if axis is None:
        axis = plt.gca()

    make_boxes_boxes(axis, xdata, ydata, xerror, yerror)


def multi_plotter(xs, thetas, name):
    for _, theta in thetas.iterrows():
        cr_predicted = sb.SigmoidBinomialPredictor.get_sigmoid_fn(theta)(xs)
        plt.plot(xs, cr_predicted, color='g', alpha=0.1)

    plt.xlabel("price");
    plt.ylabel("cr");
    plt.title(name)