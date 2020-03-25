import sys
import pymc3 as pm
import theano.tensor as tt
import numpy as np
import os
from glob import glob
import pandas as pd
from scipy.integrate import simps,cumtrapz

import matplotlib.pyplot as plt

import functools
#import warnings

import numpy as np
import george
from george.modeling import Model

from george import kernels

import emcee

import corner
import scipy.optimize as op


x_small = pd.read_table('x')
x_small = np.array(x_small)
flatx = x_small.flatten()


frequ = pd.read_table('freq.dat', sep='\s+')



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
kerns = load_kernels()



freqs_1 = np.array(frequ.loc[frequ['l'] == 1]['Freqs'])
freqs_2 = np.array(frequ.loc[frequ['l'] == 2]['Freqs'])
freqs_3 = np.array(frequ.loc[frequ['l'] == 3]['Freqs'])

split_vals_1 = np.array(frequ.loc[frequ['l'] == 1]['delta'])*1E-4
split_vals_2 = np.array(frequ.loc[frequ['l'] == 2]['delta'])*1E-4
split_vals_3 = np.array(frequ.loc[frequ['l'] == 3]['delta'])*1E-4

e_split_vals_1 = np.array(frequ.loc[frequ['l'] == 1]['e_delta'])*1E-4
e_split_vals_2 = np.array(frequ.loc[frequ['l'] == 2]['e_delta'])*1E-4
e_split_vals_3 = np.array(frequ.loc[frequ['l'] == 3]['e_delta'])*1E-4


fig = plt.figure(figsize=(12,5)); ax = fig.gca()
ax.plot(freqs_1, split_vals_1, lw=3, label="l = 1");
ax.plot(freqs_2, split_vals_2, lw=3, label="l = 2");
ax.plot(freqs_3, split_vals_3, lw=3, label="l = 3");

ax.set_xlabel("X"); ax.set_ylabel("y"); plt.legend();
plt.show()


x_diffs = np.zeros_like(flatx)
for j in range(1,len(flatx)):
    x_diffs[j] = flatx[j] - flatx[j-1]
    
    
def splittings(omega, l):
    vals = []
    for n in frequ.loc[frequ['l']==l]['n']: # 0 to 35?
        area = np.dot(
            x_diffs,
            omega * kerns[l, n, :]
        )
        beta_mask = (beta[:, 0] == l) * (beta[:, 1] == n)
        delta = beta[beta_mask, 2] * area

        vals.append(delta[0])
    vals = np.array(vals)
    return vals

class Model(Model):
    parameter_names = ("a", "b", "c", "log_sigma")

    def get_value(self, t):
        rot_prof = self.a * np.exp(-0.5*(flatx-self.b)**2 * np.exp(-2*self.log_sigma)) + self.c
        vals = splittings(rot_prof,1)
        return vals

truth = dict(a=0.4, b=10, c = 0.5, log_sigma=np.log(0.4))

kwargs = dict(**truth)
kwargs["bounds"] = dict(location=(-2, 2))
mean_model = Model(**kwargs)

gp = george.GP(0.01 * kernels.ExpSquaredKernel(10), mean=mean_model)
#this is the part where im confused
#gp.compute(flatx)

gp.compute(freqs_1, e_split_vals_1)

def lnprob2(p):
    gp.set_parameter_vector(p)
    return -gp.log_likelihood(split_vals_1, quiet=True) - gp.log_prior()
    

def grad_nll(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(split_vals_1, quiet=True)

#take a plot of initial guess
plt.figure()
plt.plot(freqs_1,split_vals_1,label = 'Real')
plt.plot(freqs_1,mean_model.get_value(1), label = 'First Guess')
plt.legend(loc = 'best')
plt.show()

print(gp.log_likelihood(split_vals_1))

p0 = gp.get_parameter_vector()
results = op.minimize(lnprob2, p0)

gp.set_parameter_vector(results.x)
print(gp.log_likelihood(split_vals_1))

plt.figure()
plt.plot(freqs_1,split_vals_1,label = 'Real')
plt.plot(freqs_1,mean_model.get_value(1), label = 'Optimized Guess')
plt.errorbar(freqs_1, split_vals_1, yerr=e_split_vals_1, fmt=".k", capsize=0)
plt.legend(loc = 'best')
plt.show()
    
initial = gp.get_parameter_vector()
ndim, nwalkers = len(initial), 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2)

print("Running first burn-in...")
p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 2000)

print("Running second burn-in...")
p0 = p0[np.argmax(lp)] + 1e-8 * np.random.randn(nwalkers, ndim)
sampler.reset()
p0, _, _ = sampler.run_mcmc(p0, 2000)
sampler.reset()

print("Running production...")
sampler.run_mcmc(p0, 2000);

# Plot the data.
plt.errorbar(freqs_1, split_vals_1, yerr=e_split_vals_1, fmt=".k", capsize=0)

# The positions where the prediction should be computed.
x = np.linspace(-5, 5, 500)

# Plot 24 posterior samples.
samples = sampler.flatchain
for s in samples[np.random.randint(len(samples), size=24)]:
    gp.set_parameter_vector(s)
    mu = gp.sample_conditional(split_vals_1,1)
    plt.plot(freqs_1, mu, color="#4682b4", alpha=0.3)

plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
plt.xlim(-5, 5)
plt.title("fit with GP noise model");
#
#names = gp.get_parameter_names()
#inds = np.array([names.index("mean:"+k) for k in tri_cols])
#corner.corner(sampler.flatchain[:, inds], truths=tri_truths, labels=tri_labels);
#

