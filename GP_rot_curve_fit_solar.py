import sys
import pymc3 as pm
import theano.tensor as tt
import numpy as np
import os
from glob import glob
import pandas as pd
from scipy.integrate import simps,cumtrapz

from astropy.table import Table

import matplotlib.pyplot as plt

# set the seed
np.random.seed(5)

n = 4800 # The number of data points
n2 = 20
X = np.linspace(0, 1, n)[:, None] # The inputs to the GP, they must be arranged as a column vector

x = np.loadtxt("x", skiprows=1)
beta_table = Table.read("beta.dat", format="ascii")
freq_table = Table.read("freq.dat", format="ascii")




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

def construct_beta_matrix(beta_table):
    ls = np.unique(beta_table["l"]).size
    ns = np.unique(beta_table["n"]).size

    beta = np.nan * np.ones((ls, ns))
    for row in beta_table:
        beta[row["l"] - 1, row["n"] - 1] = row["beta"]
    
    return beta

beta = construct_beta_matrix(beta_table)

kern = load_kernels()

freq_nl = freq_table[(27 > freq_table["n"]) * (freq_table["n"] > 8)]

freqs_1 = freq_nl["Freqs"][freq_nl["l"] == 1]*1E3
freqs_2 = freq_nl["Freqs"][freq_nl["l"] == 2]*1E3
freqs_3 = freq_nl["Freqs"][freq_nl["l"] == 3]*1E3



lone = np.ones((np.shape(freqs_1)[0],2))
ltwo = 2*np.ones((np.shape(freqs_2)[0],2))
lthree = 3*np.ones((np.shape(freqs_3)[0],2))

lone[:,0] = freqs_1
ltwo[:,0] = freqs_2
lthree[:,0] = freqs_3

xvals = np.append(lone,ltwo,axis = 0)
xvals = np.append(xvals,lthree,axis = 0)

freqs = np.array([freqs_1,freqs_2,freqs_3])

split_vals_1 = freq_nl["delta"][freq_nl["l"] == 1]
split_vals_2 = freq_nl["delta"][freq_nl["l"] == 2]
split_vals_3 = freq_nl["delta"][freq_nl["l"] == 3]


split_vals_plot = np.array([split_vals_1,split_vals_2,split_vals_3])

split_vals = np.append(split_vals_1,split_vals_2)
split_vals = np.append(split_vals, split_vals_3)



e_split_vals_1 = freq_nl["e_delta"][freq_nl["l"] == 1]
e_split_vals_2 = freq_nl["e_delta"][freq_nl["l"] == 1]
e_split_vals_3 = freq_nl["e_delta"][freq_nl["l"] == 1]

e_split_vals = np.append(e_split_vals_1,e_split_vals_2)
e_split_vals = np.append(e_split_vals, e_split_vals_3)


fig = plt.figure(figsize=(12,5)); ax = fig.gca()
ax.plot(freqs_1, split_vals_1, lw=3, label="l = 1");
ax.plot(freqs_2, split_vals_2, lw=3, label="l = 2");
ax.plot(freqs_3, split_vals_3, lw=3, label="l = 3");

ax.set_xlabel("Frequency"); ax.set_ylabel("Splitting Value"); plt.legend();
plt.show()

sol_splittings = pd.read_table('freq.dat',sep='\s+')

split_vals = np.array(sol_splittings['delta'])*1E-4

print(len(freqs),len(split_vals))


#

#
#
#
#
x_diffs = np.zeros_like(x)
for j in range(1,len(x)):
    x_diffs[j] = x[j] - x[j-1]


#
def splittings(omega, l):

    area = np.dot(x_diffs, (omega * kern[l, 10:27]).T)
    return area * beta[l - 1, 9:26]#
class CustomMean(pm.gp.mean.Mean):

    def __init__(self, a = 0, b = 1, c = 0):
        pm.gp.mean.Mean.__init__(self)
        self.a = a
        self.b = b
        self.c = c


    def __call__(self, X):
        print(len(X))
        rot_prof = tt.squeeze(self.a * tt.exp(tt.dot(-x,self.b)) + self.c)
        # Debugging
        print("rot_prof: ", rot_prof.tag.test_value)
        #return rot_prof

        # A one dimensional column vector of inputs.

        vals = splittings(rot_prof, 1)
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
