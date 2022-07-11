import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from statsmodels.tsa.stattools import adfuller
import progressbar
from time import sleep

#Exponential decay.
x = np.linspace(0, 1, 1000)
y1 = 1.0 / np.power(np.exp(x), 10)

#Grid search to look for beta values. (score: adfuller p-value, null hypothesis = stationary data)
search_size = 10
grid = np.linspace(0.001, 10, search_size)
min_test = 1000
min_a = 0
min_b = 0
total_configs = search_size*search_size
count_configs = 0
bar = progressbar.ProgressBar(maxval=total_configs, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
print("Performing grid search of beta hyper-parameters")
for i in grid:
    for j in grid:
        beta_1 = beta.cdf(x, i, j)
        z = y1 * beta_1
        stat_coef = adfuller(z)[0] #p-value associated, the lower the better.
        if stat_coef < min_test:
            min_a = i
            min_b = i
            min_test = stat_coef
        count_configs = count_configs + 1
        bar.update(count_configs)

#Exponential decay figure.
y1 = 1.0/np.power(np.exp(x), 10)
f = plt.figure()
plt.plot(x, y1)
plt.xlabel("Input space")
plt.ylabel("Non stationary function")
plt.title("Exponential decay")
plt.show()
f.savefig("beta1.pdf", bbox_inches='tight')

#Beta max figure.
beta_1 = beta.cdf(x, min_a, min_b)
f = plt.figure()
plt.plot(x, beta_1)
plt.xlabel("Input space")
plt.ylabel("Beta distribution")
plt.title("Beta dist. that adjusts exponential decay")
plt.show()
f.savefig("beta3.pdf", bbox_inches='tight')

#Post warping.
psa = y1 * beta.cdf(x, min_a, min_b)
f = plt.figure()
plt.plot(x, psa)
plt.xlabel("Input space")
plt.ylabel("Beta distribution")
plt.title("Post warping exponential decay")
plt.show()
f.savefig("beta5.pdf", bbox_inches='tight')
