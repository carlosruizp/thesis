import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t = np.arange(0.0, 1.0, 0.005)
s = np.sin(4 * np.pi * t) + 1
z = np.sum(s)
s = s/z

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='Input space', ylabel='p(x)',
       title='Slice sampling over target distribution')
ax.grid()

fig.savefig('sin.pdf')  
plt.show()
