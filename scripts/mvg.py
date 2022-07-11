import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

x, y = np.mgrid[-1:1:.005, -1:1:.005]
pos = np.dstack((x, y))
rv = multivariate_normal([-0.7, -0.7], [[0.02, 0], [0.0, 0.02]])
m2 = np.array([0.5, 0.2])
m3 = np.array([-0.5, 0.2])
s2 = np.array([[0.03, 0.05],[0.01, 0.15]])
s3 = np.array([[0.03, 0.05],[0.01, 0.15]])
rv2 = multivariate_normal(m2, s2)
rv3 = multivariate_normal(m3, s3)
s4 = np.linalg.inv(np.linalg.inv(s2) + np.linalg.inv(s3))
m4 = np.matmul(np.matmul(s4, np.linalg.inv(s2)), m2) + np.matmul(np.matmul(s4, np.linalg.inv(s3)), m3)
s5 = np.array([[0.2, 0.05],[0.01, 0.15]])
rv4 = multivariate_normal(m4, s5)
c1 = plt.contour(x, y, rv.pdf(pos), colors='red', label='Proposal distribution')
c2 = plt.contour(x, y, rv2.pdf(pos) + rv3.pdf(pos), colors='blue', label='Target distribution')
c3 = plt.contour(x, y, rv4.pdf(pos), colors='brown', label='Optimal distribution of the exponential family')
h1, _ = c1.legend_elements()
h2, _ = c2.legend_elements()
h3, _ = c3.legend_elements()
plt.title('Moment matching of distributions')
plt.legend([h1[0],h2[0],h3[0]], ['Proposal distribution', 'Target distribution', 'Exponential family optimal distribution'], loc='lower right')
plt.savefig('mvg.pdf')
plt.show()
