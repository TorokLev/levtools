from matplotlib import pylab as plt
import numpy as np
from sklearn.decomposition import PCA


def pca_plot(x, y):
    pca = PCA(n_components=2).fit(np.array([x, y]).T)
    plt.plot(x, y, '.', alpha=.3, label='samples')

    print("components: ", pca.components_)
    print("Explained variance: ", pca.explained_variance_)

    x_mean = x.mean()
    y_mean = y.mean()
    for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
        comp = comp * np.sqrt(var)  # scale component by its std explanation power
        plt.plot([x_mean, x_mean + comp[0]],
                 [y_mean, y_mean + comp[1]], label=f"Component {i}", linewidth=3, color=f"C{i + 2}")
    plt.legend()
    plt.show()

    return pca
