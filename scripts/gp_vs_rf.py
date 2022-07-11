import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

GRID_SIZE = 1000

def objective_function(x):
    return np.exp(np.sin(25*x) + np.cos(15*x))

if __name__ == '__main__':
    x = np.linspace(0, 1, GRID_SIZE)
    obj = objective_function(x)
    n_samples = 10
    x_train = np.random.choice(x, n_samples).reshape((n_samples, 1))
    y_train = objective_function(x_train).ravel()
    rf = RandomForestRegressor(max_depth=20, random_state=0)
    rf.fit(x_train, y_train)
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp.fit(x_train, y_train) 
    y_pred_gp, sigma_gp = gp.predict(x.reshape((GRID_SIZE, 1)), return_std=True)
    y_pred_rf, sigma_rf = rf.predict(x.reshape((GRID_SIZE, 1)))
    plt.plot(x, obj, 'r', label='Ground truth')
    plt.plot(x, y_pred_gp, 'g', label='GP prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred_gp+sigma_gp, y_pred_gp[::-1]-sigma_gp[::-1]]),
         alpha=.5, fc='b', ec='None', label='GP uncertainty')
    plt.plot(x_train, y_train, 'r.', markersize=10, label='Observations')
    plt.xlabel('Input space')
    plt.ylabel('Output space')
    plt.title('GP Regression')
    plt.ylim(-6, 10)
    plt.legend()
    plt.show()
    plt.plot(x, obj, 'r', label='Ground truth')
    plt.plot(x, y_pred_rf, 'g', label='RF prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred_rf+sigma_rf, y_pred_rf[::-1]-sigma_rf[::-1]]),
         alpha=.5, fc='b', ec='None', label='GP uncertainty')
    plt.plot(x_train, y_train, 'r.', markersize=10, label='Observations')
    plt.xlabel('Input space')
    plt.ylabel('Output space')
    plt.title('RF Regression')
    plt.ylim(-6, 10)
    plt.legend()
    plt.show()
