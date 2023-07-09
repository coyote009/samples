import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp_stats

N = 1000
std = 1

mu0 = 0
mu1 = 0.1

d0s = np.random.normal(mu0, std, size=(N,))
d1s = np.random.normal(mu1, std, size=(N,))

plt.plot(d0s, "o-", label="d0")
plt.plot(d1s, "o-", label="d1")
plt.grid()
plt.show()

diffs = d1s - d0s
mean = np.var(diffs)
var = np.var(diffs, ddof=1)

t_val = mean / np.sqrt(var/N)

ttest = sp_stats.ttest_rel(d0s, d1s)
