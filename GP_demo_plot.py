import numpy as np
import matplotlib.pyplot as plt
import george
from george.kernels import ExpSquaredKernel
import fitsio
import glob
from plotstuff import params, colours
cols = colours()
params()

# Load the data
kid = "008311864"
fnames = \
    glob.glob("/Users/ruthangus/.kplr/data/lightcurves/{0}/*".format(kid))
x, y, yerr = [], [], []
for fname in fnames:
    data = fitsio.read(fname)
    time = data["TIME"]
    flux = data["PDCSAP_FLUX"]
    err = data["PDCSAP_FLUX_ERR"]
    m = np.isfinite(time) * np.isfinite(flux) * np.isfinite(err)
    x.append(time[m])
    med = np.median(flux[m])
    y.append(flux[m]/med - 1)
    yerr.append(err[m]/med)
x = [i for j in x for i in j]
y = [i for j in y for i in j]
yerr = [i for j in yerr for i in j]
x -= x[0]
m1, m2 = 2100, 7000
x, y, yerr = x[m1:m2], y[m1:m2], yerr[m1:m2]
x -= x[0]

# fit a GP
A, l = 100, 5
K = A**2 * ExpSquaredKernel(l**2)
gp = george.GP(K)
gp.compute(x, yerr)
# gp.optimize(x, y, yerr)
xs = np.linspace(min(x), max(x), 1000)
mu, cov = gp.predict(y, xs)

# plot the data
plt.clf()
plt.plot(x, y, "k.")
# plt.plot(xs, mu, color=cols.lightblue, lw=2)
plt.plot(xs, mu, color=cols.orange, lw=2)
plt.xlabel("$\mathrm{Time~(Days)}$")
plt.ylabel("$\mathrm{Normalised~flux}$")
plt.subplots_adjust(left=.2)
plt.xlim(0, max(x))
plt.ylim(-.0008, .0008)
plt.savefig("Kepler452b")
