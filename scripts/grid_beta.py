import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

x = np.linspace(0,1,1000)
target = -x + 1
min_discrepancy = 1000000
adef = 0
bdef = 0
for i in range(10000):
    a1 = np.random.rand() + 1
    b1 = np.random.rand()
    psa = 1.0/np.power(np.exp(beta.cdf(x, a1, b1)), 10)
    actual = np.sum(np.abs(target - psa))
    if actual < min_discrepancy:
        adef = a1
        bdef = b1
        min_discrepancy = actual

psa = 1.0/np.power(np.exp(beta.cdf(x, adef, bdef)), 10)
f = plt.figure()
plt.plot(x, psa)
plt.xlabel("Input space")
plt.ylabel("Beta distribution")
plt.title("Post warping exponential decay")
plt.show()
f.savefig("beta_exponential_post.pdf", bbox_inches='tight')
print(adef)
print(bdef)

y1 = 1.0/np.power(np.exp(x), 10)
f = plt.figure()
plt.plot(x, y1)
plt.xlabel("Input space")
plt.ylabel("Non stationary function")
plt.title("Exponential decay")
plt.show()
f.savefig("beta_exponential.pdf", bbox_inches='tight')

beta_1 = beta.cdf(x, adef, bdef)
f = plt.figure()
plt.plot(x, beta_1)
plt.xlabel("Input space")
plt.ylabel("Beta distribution")
plt.title("Beta dist. that adjusts exponential decay")
plt.show()
f.savefig("beta_exponential_beta.pdf", bbox_inches='tight')
