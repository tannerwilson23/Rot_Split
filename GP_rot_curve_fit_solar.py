import sys
import pymc3 as pm
import theano.tensor as tt
import numpy as np
import os
from glob import glob
import pandas as pd
from scipy.integrate import simps,cumtrapz

import matplotlib.pyplot as plt

# set the seed
np.random.seed(5)

n = 4800 # The number of data points
n2 = 20
X = np.linspace(0, 1, n)[:, None] # The inputs to the GP, they must be arranged as a column vector

x_small = pd.read_table('x')
x_small = np.array(x_small)
flatx = x_small.flatten()


frequ = pd.read_table('fgong-freqs.dat', sep='\s+')


# Load stuff once.
def load_kernels(dirname="kerns/", size=4800):
    paths = glob(dirname + "/l.*_n.*")
    max_l = 1 + max([int(path.split(".")[1].split("_")[0]) for path in paths])
    max_n = 1 + max([int(path.split(".")[-1]) for path in paths])

    kerns = np.nan * np.ones((max_l, max_n, size))

    for l in range(1 + max_l):
        for n in range(1 + max_n):
            path = os.path.join(dirname, "l.{l:.0f}_n.{n:.0f}".format(l=l, n=n))
            if not os.path.exists(path): continue
            kerns[l, n, :] = np.loadtxt(path, skiprows=1)

    return kerns

beta = np.loadtxt("beta.dat", skiprows=1) # l, n, beta
kernels = load_kernels()



#def splittings_basic(omega, x, lval):
#    for l in [lval]:
#        vals = []
#        freqs = np.array(frequ.loc[frequ['l'] == l]['nu'])
#        for i in range(6,28):
#                delt = simps(omega*kernels[l,i,:], x)
#                beta_mask = (beta[:, 0] == l) * (beta[:, 1] == i)
#                if (i == 1):
#                    print(beta[beta_mask, 2])
#                delta = beta[beta_mask, 2] * delt
#                vals.append(delta[0])
#    np.array(vals)
#    return freqs, vals

freqs = np.array(frequ.loc[frequ['l'] == 1]['nu'])[6:28]
#freqs,vals = splittings_basic(f_true, flatx, 1)

freqs = freqs[:, None]



sol_splittings = pd.read_table('sol_splittings.dat',sep='\s+')

split_vals = np.array(sol_splittings['delta'])*1E-4

print(len(freqs),len(split_vals))

## Plot the data and the unobserved latent function
fig = plt.figure(figsize=(12,5)); ax = fig.gca()
ax.plot(freqs, split_vals, "dodgerblue", lw=3, label="True f");

ax.set_xlabel("X"); ax.set_ylabel("y"); plt.legend();
plt.show()
#

#
#
#
#
x_diffs = np.zeros_like(flatx)
for j in range(1,len(flatx)):
    x_diffs[j] = flatx[j] - flatx[j-1]


#
def splittings(omega, x, l):
    vals = []
    for n in range(6, 28): # 0 to 35?
        area = tt.dot(
            x_diffs,
            omega * kernels[l, n, :]
        )
        beta_mask = (beta[:, 0] == l) * (beta[:, 1] == n)
        delta = beta[beta_mask, 2] * area

        vals.append(delta)
    vals = tt.as_tensor_variable(vals)
    vals = tt.squeeze(vals)
    print("vals")
    print(vals.tag.test_value)
    return vals
#
#
class CustomMean(pm.gp.mean.Mean):

    def __init__(self, a = 0, b = 1, c = 0):
        pm.gp.mean.Mean.__init__(self)
        self.a = a
        self.b = b
        self.c = c


    def __call__(self, X):
        print(len(X))
        rot_prof = tt.squeeze(self.a * tt.exp(tt.dot(-flatx,self.b)) + self.c)
        # Debugging
        print("rot_prof: ", rot_prof.tag.test_value)
        #return rot_prof

        # A one dimensional column vector of inputs.

        vals = splittings(rot_prof, flatx, 1)
        print("outer vals", vals.tag.test_value)
        return vals
#
#
with pm.Model() as model:

    ℓ = pm.Gamma("ℓ", alpha=2, beta=4)
    η = pm.HalfNormal("η", sd=1.0)

    cov_trend = η**2 * pm.gp.cov.ExpQuad(1, ℓ)


    a_var = pm.Normal("a_var", mu = 0., sd=10)
    b_var = pm.Normal("b_var", mu = 0., sd=20)
    c_var = pm.Normal("c_var", mu = 0., sd=10)


    mean_trend = CustomMean(a = a_var, b= b_var, c= c_var)

    gp_trend = pm.gp.Latent(mean_func = mean_trend , cov_func=cov_trend)
    f = gp_trend.prior("f", X = freqs)

    # The Gaussian process is a sum of these three components
    σ  = pm.HalfNormal("σ",  sd=2.0)

    y_ = pm.StudentT('y', mu = f, sd = σ,nu = 1, observed = split_vals)
    #     y_ = pm.Normal('y', mu = f, sigma = σ, observed = y)

    # this line calls an optimizer to find the MAP
    #mp = pm.find_MAP(include_transformed=True)


    trace = pm.sample(1000, chains=1)





fig = plt.figure(figsize=(12,5)); ax = fig.gca()
# plot the samples from the gp posterior with samples and shading
from pymc3.gp.util import plot_gp_dist
plot_gp_dist(ax, trace["f"], freqs);

# plot the data and the true latent function
ax.plot(freqs, split_vals, "dodgerblue", lw=3, label="True f");

ax.set_xlabel("X"); ax.set_ylabel("y"); plt.legend();
# axis labels and title
plt.xlabel("X"); plt.ylabel("True f(x)");
plt.title("Posterior distribution over $f(x)$ at the observed values"); plt.legend();

#create the posterior/trace plots of the variables.
lines = [
    ("η",  {}, 0.001),
    ("σ", {}, 0.003),
    ("ℓ", {}, 0.05),
    ("a_var", {}, 0.4),
    ("b_var", {}, 10),
    ("c_var", {}, 0.4),
]
pm.traceplot(trace)
# lines=lines, varnames=["η", "σ", "ℓ", "a_var","b_var","c_var"]);
