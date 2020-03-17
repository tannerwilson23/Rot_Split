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

flatx_ten = np.array([flatx,flatx,flatx])


frequ = pd.read_table('freq.dat', sep='\s+')


#load in all of the possible n and l kernels and the associated frequencies

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


###instead of just some x we need to use the acoustic depth
###



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



#load in the correct frequencies
freqs_1 = np.array(frequ.loc[frequ['l'] == 1]['Freqs'])
freqs_2 = np.array(frequ.loc[frequ['l'] == 2]['Freqs'])
freqs_3 = np.array(frequ.loc[frequ['l'] == 3]['Freqs'])

#freqs,vals = splittings_basic(f_true, flatx, 1)
freqs_1 = freqs_1[:, None]
freqs_2 = freqs_2[:, None]
freqs_3 = freqs_3[:, None]

freqs = np.concatenate((freqs_1,freqs_2,freqs_3))

#load in all of the solar splittings


split_vals_1 = np.array(frequ.loc[frequ['l'] == 1]['delta'])*1E-4
split_vals_2 = np.array(frequ.loc[frequ['l'] == 2]['delta'])*1E-4
split_vals_3 = np.array(frequ.loc[frequ['l'] == 3]['delta'])*1E-4
split_vals = np.concatenate((split_vals_1,split_vals_2,split_vals_3))

## Plot the data and the unobserved latent function
fig = plt.figure(figsize=(12,5)); ax = fig.gca()
ax.plot(freqs_1, split_vals_1, lw=3, label="l = 1");
ax.plot(freqs_2, split_vals_2, lw=3, label="l = 2");
ax.plot(freqs_3, split_vals_3, lw=3, label="l = 3");

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




#need to change here to be able to take in the range of the n values for the specific l
#need to spit out all of the possible frequencies and the splitting vlaues and these
#need to be the same size
def splittings(omega, x, l):
    vals = []
    for n in frequ.loc[frequ['l']==l]['n']: # 0 to 35?
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
        rot_prof_0 = tt.squeeze(self.a[0] * tt.exp(tt.dot(-flatx,self.b[0])) + self.c[0])
        rot_prof_1 = tt.squeeze(self.a[1] * tt.exp(tt.dot(-flatx,self.b[1])) + self.c[1])
        rot_prof_2 = tt.squeeze(self.a[2] * tt.exp(tt.dot(-flatx,self.b[2])) + self.c[2])
        # Debugging
        print("rot_prof: ", rot_prof_0.tag.test_value)
        print("rot_prof: ", rot_prof_1.tag.test_value)
        print("rot_prof: ", rot_prof_2.tag.test_value)
        #return rot_prof

        # A one dimensional column vector of inputs.

        vals_1 = splittings(rot_prof_0, flatx, 1)
        vals_2 = splittings(rot_prof_1, flatx, 2)
        vals_3 = splittings(rot_prof_2, flatx, 3)
        vals_full = tt.concatenate((vals_1,vals_2,vals_3))
        return vals_full
#
#
with pm.Model() as model:

    ℓ = pm.Gamma("ℓ", alpha=2, beta=4)
    η = pm.HalfNormal("η", sd=1.0)

    cov_trend = η**2 * pm.gp.cov.ExpQuad(1, ℓ)

    mu_a = pm.Normal('mu_a', mu=0., sigma=100)
    sigma_a = pm.HalfNormal('sigma_a', 5.)
    mu_b = pm.Normal('mu_b', mu=0., sigma=100)
    sigma_b = pm.HalfNormal('sigma_b', 5.)
    mu_c = pm.Normal('mu_c', mu=0., sigma=100)
    sigma_c = pm.HalfNormal('sigma_c', 5.)



    a_var = pm.Normal("a_var", mu = mu_a, sigma=sigma_a, shape = 3)
    b_var = pm.Normal("b_var", mu = mu_b, sigma=sigma_b, shape = 3)
    c_var = pm.Normal("c_var", mu = mu_c, sigma=sigma_c, shape = 3)


    mean_trend = CustomMean(a = a_var, b= b_var, c= c_var)

    gp_trend = pm.gp.Latent(mean_func = mean_trend , cov_func=cov_trend)
    f = gp_trend.prior("f", X = freqs)

    # The Gaussian process is a sum of these three components
    σ  = pm.HalfNormal("σ",  sd=10.0)

    y_ = pm.Normal('y', mu = f, sigma= σ, observed=split_vals)
    #     y_ = pm.Normal('y', mu = f, sigma = σ, observed = y)

    # this line calls an optimizer to find the MAP
    #mp = pm.find_MAP(include_transformed=True)


    trace = pm.sample(2000, tune = 1000, chains=1)





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

##create the posterior/trace plots of the variables.
#lines = [
#    ("η",  {}, 0.001),
#    ("σ", {}, 0.003),
#    ("ℓ", {}, 0.05),
#    ("a_var", {}, 0.4),
#    ("b_var", {}, 10),
#    ("c_var", {}, 0.4),
#]
#pm.traceplot(trace)
## lines=lines, varnames=["η", "σ", "ℓ", "a_var","b_var","c_var"]);
