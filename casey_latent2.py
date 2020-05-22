
import os
import numpy as np
from scipy import (integrate, optimize as op)
from astropy import units as u
from astropy.table import Table
from glob import glob
import theano.tensor as tt

import matplotlib.pyplot as plt
import pymc3 as pm


# Generate some true profile and splittings.


x = np.loadtxt("x", skiprows=1)
N = 10 # the number of points you think you can afford in your GP
S = int(x.size / N) # approx sampling size.

x = x[::S]
x_diffs = np.hstack([0, np.diff(x)])


beta_table = Table.read("beta.dat", format="ascii")
freq_table = Table.read("freq.dat", format="ascii")


# TODO: why we do this?
freq_nl = freq_table[(27 > freq_table["n"]) * (freq_table["n"] > 9)]


l1_frequencies = np.array(freq_nl["Freqs"][freq_nl["l"] == 1])
l2_frequencies = np.array(freq_nl["Freqs"][freq_nl["l"] == 2])
l3_frequencies = np.array(freq_nl["Freqs"][freq_nl["l"] == 3])


def get_omega(a, b, c, d):
    x_offset = x - 0.5
    return a * np.arctan((x_offset - c) * b) + d


def load_kernels(dirname="kerns/", size=4800):
    paths = glob(dirname + "/l.*_n.*")
    max_l = 1 + max([int(path.split(".")[1].split("_")[0]) for path in paths])
    max_n = 1 + max([int(path.split(".")[-1]) for path in paths])

    kerns = np.nan * np.ones((max_l, max_n, size))

    for l in range(1 + max_l):
        for n in range(1 + max_n):
            path = os.path.join(dirname, "l.{l:.0f}_n.{n:.0f}".format(l=l, n=n))
            if not os.path.exists(path): 
                # TODO: some warning?
                continue
            
            kerns[l, n, :] = np.loadtxt(path, skiprows=1)

    return kerns

kern = load_kernels()


modelS = Table.read('modelS.dat', format="ascii")
# acoustic depth: tau(r) = int_r^R 1/c(x) dx (Aerts et al. eqn. 3.228)
c2 = np.sqrt(modelS['c2'])
r  = modelS['r']
tau = np.hstack([0, -integrate.cumtrapz(1/c2, r)])

x_diffs = np.hstack([0, np.diff(x)])

def construct_beta_matrix(beta_table):
    ls = np.unique(beta_table["l"]).size
    ns = np.unique(beta_table["n"]).size

    beta = np.nan * np.ones((ls, ns))
    for row in beta_table:
        beta[row["l"] - 1, row["n"] - 1] = row["beta"]
    
    return beta

beta = construct_beta_matrix(beta_table)

def splittings(omega, l):
    vals = []
    for n in freq_nl["n"][freq_nl["l"] == l]:
        area = np.dot(
            x_diffs,
            omega * kern[l, n, ::S]
        )
        beta_mask = (beta_table["l"] == l) * (beta_table["n"] == n)
        delta = beta_table["beta"][beta_mask] * area
    
        vals.append(delta[0])

    # TODO: return corresponding n values? or frequencies?
    return np.array(vals)


def get_splittings(omega, l):
    area = np.dot(np.atleast_2d(x_diffs), (omega * kern[l, 10:27, ::S]).T)
    return area * beta[l - 1, 9:26]


def get_model_splittings(a, b, c, d):
    omega = get_omega(a, b, c, d)

    return [get_splittings(omega, l) for l in (1, 2, 3)]


true_params_vector = np.array([20, 70, -0.2, 440])
true_omega = get_omega(*true_params_vector)
true_omega = 450 * np.ones_like(true_omega)

fig, ax = plt.subplots()
ax.plot(x, true_omega)

true_l1_splittings, true_l2_splittings, true_l3_splittings = get_model_splittings(*true_params_vector)

scale = 10
l1_splittings = true_l1_splittings + scale * np.random.normal(0, 1, size=true_l1_splittings.size)
l2_splittings = true_l2_splittings + scale * np.random.normal(0, 1, size=true_l2_splittings.size)
l3_splittings = true_l3_splittings + scale * np.random.normal(0, 1, size=true_l3_splittings.size)

e_l1_splittings = np.abs(scale * np.random.normal(0, 1, size=true_l1_splittings.size))
e_l2_splittings = np.abs(scale * np.random.normal(0, 1, size=true_l2_splittings.size))
e_l3_splittings = np.abs(scale * np.random.normal(0, 1, size=true_l3_splittings.size))


# How many can we afford?
x = np.loadtxt("x", skiprows=1)

N = 10 # the number of points you think you can afford in your GP
S = int(x.size / N) # approx sampling size.

x = x[::S]
x_diffs = np.hstack([0, np.diff(x)])

def pm_splittings(omega, l):
    #area = np.dot(np.atleast_2d(x_diffs), (omega * kern[l, 10:27, ::S]).T)
    area = tt.dot(x_diffs, (omega * kern[l, 10:27, ::S]).T) 
    return area * beta[l - 1, 9:26]

def get_splittings(omega, l):
    area = np.dot(np.atleast_2d(x_diffs), (omega * kern[l, 10:27, ::S]).T)
    return area * beta[l - 1, 9:26]


# A one dimensional column vector of inputs.
#X = x.reshape((-1, 1))

N_latent = 10
S_latent = int(x.size / N_latent)
X_latent = x[::S_latent, None]

X = x.reshape((-1, 1))

with pm.Model() as latent_gp_model:

    mu = pm.Uniform("mu_omega", 400, 800)
    cov_func = pm.gp.cov.ExpQuad(1, ls=0.1)
    #cov_func = pm.gp.cov.Matern32(1, ls=0.2)
    gp = pm.gp.Latent(
        mean_func=pm.gp.mean.Constant(c=mu),
        cov_func=cov_func
    )


    # Place a GP prior over the function f.
    omega = gp.prior("omega", X=X_latent)

    l1_ = pm.Normal(
        'l1', 
        mu=pm_splittings(omega, 1), 
        sd=e_l1_splittings,
        observed=l1_splittings
    )

    l2_ = pm.Normal(
        'l2',
        mu=pm_splittings(omega, 2),
        sd=e_l2_splittings,
        observed=l2_splittings
    )

    l3_ = pm.Normal(
        'l3',
        mu=pm_splittings(omega, 3),
        sd=e_l3_splittings,
        observed=l3_splittings
    )

    trace = pm.sample(10000, tune=5000, chains=1)
    

fig = pm.traceplot(trace)


N_samples = 1000
with latent_gp_model:
    pred_samples = pm.sample_ppc(trace, vars=(omega, l1_, l2_, l3_), samples=N_samples)

fig, ax = plt.subplots()
for sample in pred_samples["omega"]:
    ax.plot(X.T[0], sample, c='k', alpha=0.1)

ax.plot(np.linspace(0, 1, true_omega.size), true_omega, c='tab:red', zorder=10)



colors = ("tab:blue", "tab:red", "tab:green")
fig, axes = plt.subplots(3, 1)
for l in (1, 2, 3):
    x_ = [l1_frequencies, l2_frequencies, l3_frequencies][l - 1]
    for i in range(N_samples):
        color = colors[l - 1]

        axes[l-1].plot(
            x_,
            get_splittings(true_omega[::S], l)[0],
            c=color,
            ls=":",
            lw=2
        )

        try:
            axes[l - 1].plot(
                x_,
                pred_samples["l{}".format(l)][i].flatten(),
                c=color,
                alpha=0.1
            )

        except:
            break

axes[0].scatter(
    l1_frequencies, 
    l1_splittings[0], 
    c='k', 
    zorder=10,
    s=50
)
axes[0].errorbar(
    l1_frequencies, 
    l1_splittings[0], 
    yerr=e_l1_splittings, 
    fmt='none',
    ms=10,
    zorder=10, 
    c='k'
)

 
axes[1].scatter(
    l2_frequencies, 
    l2_splittings[0], 
    c='k', 
    zorder=10,
    s=50
)
axes[1].errorbar(
    l2_frequencies, 
    l2_splittings[0], 
    yerr=e_l2_splittings, 
    fmt='none',
    ms=10,
    zorder=10, 
    c='k'
)

 
axes[2].scatter(
    l3_frequencies, 
    l3_splittings[0], 
    c='k', 
    zorder=10,
    s=50
)
axes[2].errorbar(
    l3_frequencies, 
    l3_splittings[0], 
    yerr=e_l3_splittings, 
    fmt='none',
    ms=10,
    zorder=10, 
    c='k'
)

