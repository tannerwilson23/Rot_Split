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


freqnu = frequ.loc[frequ['n'] > 9]
freqnl = freqnu.loc[freqnu['n'] < 27]


freqs_1 = np.array(freqnl.loc[freqnl['l'] == 1]['Freqs'])*1E3
freqs_2 = np.array(freqnl.loc[freqnl['l'] == 2]['Freqs'])*1E3
freqs_3 = np.array(freqnl.loc[freqnl['l'] == 3]['Freqs'])*1E3


lone = np.ones((np.shape(freqs_1)[0],2))
ltwo = 2*np.ones((np.shape(freqs_2)[0],2))
lthree = 3*np.ones((np.shape(freqs_3)[0],2))

lone[:,0] = freqs_1
ltwo[:,0] = freqs_2
lthree[:,0] = freqs_3

xvals = np.append(lone,ltwo,axis = 0)
xvals = np.append(xvals,lthree,axis = 0)

freqs = np.array([freqs_1,freqs_2,freqs_3])

split_vals_1 = np.array(freqnl.loc[freqnl['l'] == 1]['delta'])
split_vals_2 = np.array(freqnl.loc[freqnl['l'] == 2]['delta'])
split_vals_3 = np.array(freqnl.loc[freqnl['l'] == 3]['delta'])


split_vals_plot = np.array([split_vals_1,split_vals_2,split_vals_3])

split_vals = np.append(split_vals_1,split_vals_2)
split_vals = np.append(split_vals, split_vals_3)



e_split_vals_1 = np.array(freqnl.loc[freqnl['l'] == 1]['e_delta'])
e_split_vals_2 = np.array(freqnl.loc[freqnl['l'] == 2]['e_delta'])
e_split_vals_3 = np.array(freqnl.loc[freqnl['l'] == 3]['e_delta'])

e_split_vals = np.append(e_split_vals_1,e_split_vals_2)
e_split_vals = np.append(e_split_vals, e_split_vals_3)


fig = plt.figure(figsize=(12,5)); ax = fig.gca()
ax.plot(freqs_1, split_vals_1, lw=3, label="l = 1");
ax.plot(freqs_2, split_vals_2, lw=3, label="l = 2");
ax.plot(freqs_3, split_vals_3, lw=3, label="l = 3");

ax.set_xlabel("Frequency"); ax.set_ylabel("Splitting Value"); plt.legend();
plt.show()


x_diffs = np.zeros_like(flatx)
for j in range(1,len(flatx)):
    x_diffs[j] = flatx[j] - flatx[j-1]
    
    
def splittings(omega, l):
    vals = []
    for n in freqnl.loc[freqnl['l']==l]['n']: # 0 to 35?
        area = np.dot(
            x_diffs,
            omega * kerns[l, n, :]
        )
        beta_mask = (beta[:, 0] == l) * (beta[:, 1] == n)
        delta = beta[beta_mask, 2] * area
    
        vals.append(delta[0])
    vals = np.array(vals)
#    if (l == 1):
#        vals = vals
#    else if (l == 2):
#        vals = 1.5*vals
#    else:
#        vals = 2*vals
    return vals

class Model(Model):
    parameter_names = ("a", "b","c","d")

    def get_value(self, t):
        rot_prof = self.a * np.exp((-flatx*self.b-self.c/self.b))+self.d
        vals = np.append(splittings(rot_prof,1),splittings(rot_prof,2))
        vals = np.append(vals,splittings(rot_prof,3))
        return vals

    def get_value_plot(self, t):
        rot_prof = self.a * np.exp((-flatx*self.b-self.c/self.b))+self.d
        vals = np.array([splittings(rot_prof,1),splittings(rot_prof,2),splittings(rot_prof,3)])
        return vals

truth = dict(a=10, b=5, c=0.8,  d= 400)#log_sigma=np.log(0.4))

kwargs = dict(**truth)
kwargs["bounds"] = dict(a=(0,500),b = (0,50),c = (0,3),d = (200,500))
#kwargs["bounds"] = dict(a=(-5,5), b = (0,15),c = (0,10.))
mean_model = Model(**kwargs)

plt.figure()
actual_rot = 330*np.ones(4800)
actual_rot[3360:-1] = actual_rot[3360:-1] + 30

actual_1 = splittings(actual_rot,1)
actual_2 = splittings(actual_rot,2)
actual_3 = splittings(actual_rot,3)


plt.plot(freqs_1,actual_1)
plt.plot(freqs_2,actual_2)
plt.plot(freqs_3,actual_3)

plt.plot(freqs[0],split_vals_plot[0],label = 'Real', color = 'red')
#plt.plot(freqs[0],mean_model.get_value_plot(None)[0], label = 'First Guess', color = 'red', linestyle = '-.')
plt.plot(freqs[1],split_vals_plot[1],label = 'Real', color = 'blue')
#plt.plot(freqs[1],mean_model.get_value_plot(None)[1], label = 'First Guess',color = 'red', linestyle = '-.')
plt.plot(freqs[2],split_vals_plot[2],label = 'Real', color = 'green')
#plt.plot(freqs[2],mean_model.get_value_plot(None)[2], label = 'First Guess', color = 'green', linestyle = '-.')
plt.legend(loc = 'best')
plt.title('Actual Rotation Profile splitting values')

plt.show()

#gp = george.GP(0.001 * kernels.ExpSquaredKernel(metric=([1,0],[0,1]), metric_bounds = [(0,7),(None,None),(-5,-2)], ndim = 2), mean=mean_model)

#gp = george.GP(1 * kernels.ExpSquaredKernel(metric=np.array([[1,0],[0,100]]), metric_bounds=[(-5, 5), (None, None), (-5, -2.3)], ndim=2), mean=mean_model)


gp = george.GP(1 * kernels.ExpSquaredKernel(metric=1e-3 * np.eye(2), ndim=2, metric_bounds=[(-5, 5), (None, None), (-5, -2.3)]), mean = mean_model)

#gp = george.GP(1 * kernels.ExpSquaredKernel(metric=1e-5 * np.eye(2), ndim=2, metric_bounds=[(-10, 10), (None, None), (-6, -2.3)]))


gp.set_parameter('kernel:k2:metric:L_0_1',0)
gp.freeze_parameter('kernel:k2:metric:L_0_1')

#this is the part where im confused
#gp.compute(flatx)



gp.compute(xvals, e_split_vals)

def log_prior(p):
    a, b, c,d,k1_log_constant, k2_M_0, k2_M_1 = p
    #k1_log_constant, k2_M_0, k2_M_1 = p

#    if not (15 >= b >= 0):
#        return -np.inf
    if not all(i > 0 for i in (a * np.exp((-flatx*b)))):
        return -np.inf
    return 0


def log_prob(p):
    gp.set_parameter_vector(p)
    return log_prior(p) \
        +   gp.log_prior() + gp.log_likelihood(split_vals, quiet=True)

def negative_log_prob(p):
    return -log_prob(p)
    

def grad_nll(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(split_vals, quiet=True)

#take a plot of initial guess
plt.figure()

plt.plot(freqs[0],split_vals_plot[0],label = 'Real', color = 'red')
plt.plot(freqs[0],mean_model.get_value_plot(None)[0], label = 'First Guess', color = 'red', linestyle = '-.')
plt.plot(freqs[1],split_vals_plot[1],label = 'Real', color = 'blue')
plt.plot(freqs[1],mean_model.get_value_plot(None)[1], label = 'First Guess',color = 'red', linestyle = '-.')
plt.plot(freqs[2],split_vals_plot[2],label = 'Real', color = 'green')
plt.plot(freqs[2],mean_model.get_value_plot(None)[2], label = 'First Guess', color = 'green', linestyle = '-.')
plt.legend(loc = 'best')
plt.title('First Guess splitting values')
plt.show()




print(gp.log_likelihood(split_vals))

p0 = gp.get_parameter_vector()
results = op.minimize(negative_log_prob, p0, method = 'Nelder-Mead', options = dict(maxiter=2000))
print(results.x)
gp.set_parameter_vector(results.x)
print(gp.log_likelihood(split_vals))





plt.figure()
plt.plot(freqs[0],split_vals_plot[0],label = 'Real', color = 'red')
plt.errorbar(freqs[0],split_vals_plot[0],yerr = e_split_vals_1, fmt=".r", capsize=0)
plt.plot(freqs[0],mean_model.get_value_plot(None)[0], label = 'Optimized Guess', color ='red', linestyle = '-.')

plt.plot(freqs[1],split_vals_plot[1],label = 'Real', color = 'blue')
plt.plot(freqs[1],mean_model.get_value_plot(None)[1], label = 'Optimized Guess', color ='blue', linestyle = '-.')
plt.errorbar(freqs[1],split_vals_plot[1],yerr = e_split_vals_2, fmt=".b", capsize=0)

plt.plot(freqs[2],split_vals_plot[2],label = 'Real', color = 'green')
plt.plot(freqs[2],mean_model.get_value_plot(None)[2], label = 'Optimized Guess', color ='green', linestyle = '-.')
plt.errorbar(freqs[2],split_vals_plot[2],yerr = e_split_vals_2, fmt=".g", capsize=0)
plt.title('optimized')
plt.legend(loc = 'best')
plt.show()






plt.figure()
ao,bo,co,do = mean_model.get_parameter_vector()
plt.plot(flatx,(ao * np.exp((-flatx*bo-co/bo))+do))
plt.title('Rotation Curve')
plt.xlabel('r/R')
plt.ylabel(r'$\Omega$')
plt.ylim(0,500)
plt.show()

#



initial = gp.get_parameter_vector()
ndim, nwalkers = len(initial), 50
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)

print("Running first burn-in...")
p0 = initial + 1e-5 * np.random.randn(nwalkers, ndim)
state = sampler.run_mcmc(p0, 5000)





# Plot the data
#plt.errorbar(freqs_1, split_vals_1, yerr=e_split_vals_1, fmt=".k", capsize=0)

# The positions where the prediction should be computed.
# ARC says: not sure what this was meant for...

x = np.linspace(1000, 3000, 5000)


# Plot 24 posterior samples.
plt.figure()
samples = sampler.flatchain
for s in samples[np.random.randint(len(samples), size=200)]:
    print(s)
    gp.set_parameter_vector(s)
    m = gp.sample_conditional(split_vals, xvals)
    print(m)
    plt.plot(freqs[0], m[0:14], color="red", alpha=0.1)
    plt.plot(freqs[1], m[14:28], color="blue", alpha=0.1)
    plt.plot(freqs[2], m[28:42], color="green", alpha=0.1)
    

plt.plot(freqs[0],split_vals_plot[0],label = 'Real', color = 'red')
plt.errorbar(freqs[0],split_vals_plot[0],yerr = e_split_vals_1, fmt=".r", capsize=0)
plt.plot(freqs[0],mean_model.get_value_plot(None)[0], label = 'Optimized Guess', color ='red', linestyle = '-.')

plt.plot(freqs[1],split_vals_plot[1],label = 'Real', color = 'blue')
plt.plot(freqs[1],mean_model.get_value_plot(None)[1], label = 'Optimized Guess', color ='blue', linestyle = '-.')
plt.errorbar(freqs[1],split_vals_plot[1],yerr = e_split_vals_2, fmt=".b", capsize=0)

plt.plot(freqs[2],split_vals_plot[2],label = 'Real', color = 'green')
plt.plot(freqs[2],mean_model.get_value_plot(None)[2], label = 'Optimized Guess', color ='green', linestyle = '-.')
plt.errorbar(freqs[2],split_vals_plot[2],yerr = e_split_vals_2, fmt=".g", capsize=0)


plt.ylabel("Splitting")
plt.xlabel("Frequency")
#plt.xlim(-5, 5)
plt.title("Sampled with GP noise model");
#
plt.show()

names = gp.get_parameter_names()
inds = np.array([names.index(k) for k in names])
corner.corner(sampler.chain[:, 2500:, :].reshape((-1, 7)), labels=names)

#names = gp.get_parameter_names()
#corner.corner(sampler.chain.reshape((-1, 3)), labels=names)

plt.figure()
fig, axs = plt.subplots(6)
fig.suptitle('Sampler Chains')
for i in range(50): axs[0].plot(sampler.chain[i,:,0],color = 'tab:blue', alpha = 0.01)
for i in range(50): axs[1].plot(sampler.chain[i,:,1],color = 'tab:blue', alpha = 0.01)
for i in range(50): axs[2].plot(sampler.chain[i,:,2],color = 'tab:blue', alpha = 0.01)
for i in range(50): axs[3].plot(sampler.chain[i,:,3],color = 'tab:blue', alpha = 0.01)
for i in range(50): axs[4].plot(sampler.chain[i,:,4],color = 'tab:blue', alpha = 0.01)
for i in range(50): axs[5].plot(sampler.chain[i,:,5],color = 'tab:blue', alpha = 0.01)
for i in range(50): axs[6].plot(sampler.chain[i,:,6],color = 'tab:blue', alpha = 0.01)

axs[0].set_ylabel('a')
axs[1].set_ylabel('b')
axs[2].set_ylabel('b')
axs[3].set_ylabel('d')
axs[4].set_ylabel('Log_constant')
axs[5].set_ylabel('log_L_0_0')
axs[6].set_ylabel('log_L_1_1')



