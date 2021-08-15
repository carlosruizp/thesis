import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from statsmodels.tsa.stattools import adfuller

#A non-stationary periodic function.
x = np.linspace(0,1,1000)
y2 = np.sin(30*np.power(x,3))
f = plt.figure()
plt.plot(x, y2)
plt.xlabel("Input space")
plt.ylabel("Non stationary function")
plt.title("Non stationary periodic function")
plt.show()
f.savefig("beta_stationary.pdf", bbox_inches='tight')

#Second beta.
a2,b2 = 0.6, 1.5
beta_2 = beta.cdf(x, a2, b2)
f = plt.figure()
plt.plot(x, beta_2)
plt.xlabel("Input space")
plt.ylabel("Beta distribution")
plt.title("Beta dist. that adjusts periodic function")
plt.show()
f.savefig("beta_stationary_beta.pdf", bbox_inches='tight')

#Post-warping B.
psb = np.sin(30*np.power(beta.cdf(x, a2, b2),3)) 
f = plt.figure()
plt.plot(x, psb)
plt.xlabel("Input space")
plt.ylabel("Beta distribution")
plt.title("Post warping periodic function")
plt.show()
f.savefig("beta_stationary_post.pdf", bbox_inches='tight')
