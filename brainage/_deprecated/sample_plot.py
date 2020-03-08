from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt


t = np.linspace(0.0, 1.0, 100)
s = np.cos(4 * np.pi * t) + 2

fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
ax.scatter(t, s, alpha=0.5)

ax.set_xlabel(r'chronological age (years)', fontsize=16)
ax.set_ylabel(r'predicted $q_{25} (years)$', fontsize=16)
ax.set_title(r'\TeX\ is Number $\displaystyle\sum_{n=1}^\infty'
             r'\frac{-e^{i\pi}}{2^n}$!', fontsize=16, color='r')
plt.show()

import numpy as np
import matplotlib.pyplot as plt


# fig = plt.figure()
# x = np.random.random(10)
# y = 2.5 * x + np.random.random(10)
# yerr = np.linspace(0.05, 0.2, 10)
#
# plt.errorbar(x, y, yerr=yerr,
#              fmt=None)
# plt.show()
