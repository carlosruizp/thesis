import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

def indicator(pos):
    i1 = np.zeros((pos.shape[0], pos.shape[1]))
    for i in range(pos.shape[0]):
        for j in range(pos.shape[1]):
            if pos[i, j, 0] >= 0 and pos[i, j, 1] >= 0:
                i1[i, j] = 1 
    return i1

if __name__ == "__main__":
    x, y = np.mgrid[-1.5:2.5:.005, -1.5:2.5:.005]
    pos = np.dstack((x, y))
    rv = multivariate_normal([0.0, 0.0], [[0.5, 0], [0.0, 0.5]])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    i = indicator(pos)
    td = rv.pdf(pos) * i
    surf = ax.plot_surface(x, y, td, cmap=plt.cm.magma)
    plt.title('Truncated bivariate Gaussian distribution')
    plt.savefig('e1.pdf')
    plt.show()
