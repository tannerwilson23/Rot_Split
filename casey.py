
import os
import numpy as np
from scipy import (integrate, optimize as op)
from astropy import units as u
from astropy.table import Table
from glob import glob

import george
from george import kernels

import matplotlib.pyplot as plt

x = np.loadtxt("x", skiprows=1)
beta_table = Table.read("beta.dat", format="ascii")
freq_table = Table.read("freq.dat", format="ascii")


# TODO: why we do this?
freq_nl = freq_table[(27 > freq_table["n"]) * (freq_table["n"] > 9)]

freqs_l1 = freq_nl["Freqs"][freq_nl["l"] == 1]
freqs_l2 = freq_nl["Freqs"][freq_nl["l"] == 2]
freqs_l3 = freq_nl["Freqs"][freq_nl["l"] == 3]


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
            omega * kern[l, n, :]
        )
        beta_mask = (beta_table["l"] == l) * (beta_table["n"] == n)
        delta = beta_table["beta"][beta_mask] * area
    
        vals.append(delta[0])

    # TODO: return corresponding n values? or frequencies?
    return np.array(vals)


def fast_splittings(omega, l):
    
    area = np.dot(x_diffs, (omega * kern[l, 10:27]).T)
    return area * beta[l - 1, 9:26]



# fake rotation profile.
truth = dict(a=-20, b=-20, c=-13,  d=440)
a, b, c, d = (-20, -20, -13, 440)

omega = a * np.arctan(x * b - c) + d


fig, ax = plt.subplots()
ax.plot(x, omega)


fig, ax = plt.subplots()
ax.plot(freqs_l1, splittings(omega, 1))
ax.plot(freqs_l2, splittings(omega, 2))
ax.plot(freqs_l3, splittings(omega, 3))


for l in (1, 2, 3):
    assert np.allclose(
        splittings(omega, l),
        fast_splittings(omega, l)
    )

"""
In [19]: %timeit splittings(omega, 1)
3.24 ms ± 102 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [20]: %timeit fast_splittings(omega, 1)
187 µs ± 12.2 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

# ~20x faster
"""
splittings = fast_splittings

# Now let's construct the model.
kernel = 10 * kernels.Matern32Kernel(10)

gp = george.GP(
    kernel,
    mean=440,
    fit_mean=True)

gp.compute(x)

fig, ax = plt.subplots()
for sample in gp.sample(size=100):
    ax.plot(x,  sample, c="tab:blue", alpha=0.5)


faux_x = np.array([0, 1])
faux_y = np.array([430, 450])

gp.compute(faux_x)
mu, var = gp.predict(faux_y, x, return_var=True, return_cov=False)

fig, ax = plt.subplots()
ax.plot(x, mu)
ax.fill_between(
    x,
    mu - np.sqrt(var),
    mu + np.sqrt(var),
    facecolor="tab:blue",
    alpha=0.3
)


# OK, approach 1: forward model, specified profile, no GP, no noise.

r = x.copy()

def get_omega(a, b, c, d):
    # TODO: this is all fucked, should include r and use *args
    r_offset = r - 0.5
    return a * np.arctan((r_offset - c) * b) + d


def get_model_splittings(a, b, c, d):
    # TODO: assuming doing l = 1, 2, 3. 
    #       for freqs_l1, freqs_l2, etc. make this more flexible

    omega = get_omega(a, b, c, d)

    return np.hstack(
        [splittings(omega, l) for l in (1, 2, 3)]
    )

bounds = [
    (0, None),
    (-100, 100),
    (-0.5, 0.5),
    (0, None)
]

def ln_prior(params):
    a, b, c, d = params

    for i, (lower_bound, upper_bound) in enumerate(bounds):
        if (lower_bound is not None and params[i] < lower_bound) \
        or (upper_bound is not None and params[i] > upper_bound):
            return -np.inf
    return 0

def ln_likelihood(params, splittings, splittings_err):
    model_splittings = get_model_splittings(*params)
    return -0.5 * np.sum((splittings - model_splittings)**2 / splittings_err**2)

def ln_probability(params, splittings, splittings_err, verbose=False):
    v = ln_prior(params) \
         + ln_likelihood(params, splittings, splittings_err)
    if verbose:
        print(params, v)
    return v
         


nlp = lambda *args, **kwds: -ln_probability(*args, verbose=True, **kwds)

true_params_vector = np.array([20, 70, -0.2, 440])
true_splittings = get_model_splittings(*true_params_vector)
op_splittings_err = 0.1 * np.random.normal(0, 1, size=true_splittings.size)
op_splittings = true_splittings \
              + np.random.randn(true_splittings.size) * op_splittings_err


args = (
    op_splittings,
    op_splittings_err
)

print("Target: {0}".format(nlp(true_params_vector, *args)))
'''
default_value = 100
finite_bounds = [(l if l is not None else  -default_value, u if u is not None else default_value) for l, u in bounds]
init = np.mean(finite_bounds, axis=1)


init = [100, 99, 0, 450]
'''

d_default = 400
best_init = None
best_ln_prob = np.inf
# Calculate on a grid using a fixed offset?
for ga in np.linspace(0, 700, 10):
    for gb in np.linspace(*bounds[1], 10):
        for gc in np.linspace(*bounds[2], 10):
            v = ln_probability((ga, gb, gc, d_default), *args)
            if best_init is None \
            or v > best_ln_prob:
                best_init = [ga, gb, gc, d_default]
                best_ln_prob = v


init = best_init

# TODO: so wrong.
init = true_params_vector

p_opt = op.minimize(
    nlp,
    init,
    args,
    method="Nelder-Mead",
    options=dict(
        maxiter=10000,
        maxfev=10000
    ),
    bounds=bounds
)
print(p_opt)



fig, ax = plt.subplots()
ax.plot(r, get_omega(*true_params_vector), c="tab:blue", label="truth")
ax.plot(r, get_omega(*init), c="tab:red", label="init")
ax.plot(r, get_omega(*p_opt.x), c="tab:green", label="opt")

ax.legend()

# *Always* check that the optimisation *thinks* it succeeded.
print(p_opt)

assert p_opt.success
#assert ln_probability(true_params_vector, *args) >= ln_probability(p_opt.x, *args)


fig, ax = plt.subplots()
op_omega = get_omega(*p_opt.x)
init_omega = get_omega(*init)
true_omega = get_omega(*true_params_vector)

fs =     np.hstack([
        freqs_l1, freqs_l2, freqs_l3
    ])

ax.scatter(
    np.hstack([
        freqs_l1, freqs_l2, freqs_l3
    ]),
    op_splittings,
    c="k"
)
ax.errorbar(
    np.hstack([
        freqs_l1, freqs_l2, freqs_l3
    ]),
    op_splittings,
    yerr=op_splittings_err,
    fmt="none",
    c="k"
)

ax.plot(freqs_l1, splittings(op_omega, 1), c="tab:green", lw=1, alpha=0.5)
ax.plot(freqs_l2, splittings(op_omega, 2), c="tab:green", lw=2, alpha=0.5)
ax.plot(freqs_l3, splittings(op_omega, 3), c="tab:green", lw=3, alpha=0.5)

ax.plot(freqs_l1, splittings(init_omega, 1), c="tab:red", lw=1, alpha=0.5)
ax.plot(freqs_l2, splittings(init_omega, 2), c="tab:red", lw=2, alpha=0.5)
ax.plot(freqs_l3, splittings(init_omega, 3), c="tab:red", lw=3, alpha=0.5)

ax.plot(freqs_l1, splittings(true_omega, 1), c="tab:blue", lw=1, alpha=0.5)
ax.plot(freqs_l2, splittings(true_omega, 2), c="tab:blue", lw=2, alpha=0.5)
ax.plot(freqs_l3, splittings(true_omega, 3), c="tab:blue", lw=3, alpha=0.5)


# Now sample. 
import emcee

initial = p_opt.x
ndim = p_opt.x.size
nwalkers = 64

sampler = emcee.EnsembleSampler(
    nwalkers, 
    ndim, 
    ln_probability,
    args=args
)

offset = p_opt.x.copy()
p0 = offset + 1e-3 * np.random.randn(nwalkers, ndim)

state = sampler.run_mcmc(p0, 10000, progress=True)

samples = sampler.get_chain()

fig, axes = plt.subplots(ndim)
for i in range(ndim):
    axes[i].plot(samples[:, :, i], c="k", alpha=0.3)

fig.savefig("casey_chains.png")

flat_samples = samples.reshape((-1, ndim))

fig, ax = plt.subplots()
for idx in np.random.choice(flat_samples.shape[0], 100):
    ax.plot(x, get_omega(*flat_samples[idx]), c="tab:blue", alpha=0.3)

ax.plot(x, true_omega, c='k', lw=2)


raise a

"""
This looks shit. try again
"""

v = np.median(samples[5000:].reshape((-1, ndim)), axis=0)
p0 = v + 1e-5 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(
    nwalkers, 
    ndim, 
    ln_probability,
    args=args
)
state = sampler.run_mcmc(p0, 5000, progress=True)

# plot chains (again)
samples = sampler.get_chain()

fig, axes = plt.subplots(ndim)
for i in range(ndim):
    axes[i].plot(samples[:, :, i], c="k", alpha=0.3)

from corner import corner

flat_samples = samples[5000:].reshape((-1, ndim))
fig = corner(
    flat_samples,
    truths=true_params_vector
)

map_value = np.median(flat_samples, axis=0)


fig, ax = plt.subplots()
ax.plot(x, get_omega(*map_value))
ax.plot(x, get_omega(*true_params_vector), c="k")


fig, ax = plt.subplots()
map_omega = get_omega(*map_value)
true_omega = get_omega(*true_params_vector)

fs =     np.hstack([
        freqs_l1, freqs_l2, freqs_l3
    ])

ax.scatter(
    fs,
    op_splittings,
    c="k"
)
ax.errorbar(
    fs,
    op_splittings,
    yerr=op_splittings_err,
    fmt="none",
    c="k"
)

ax.plot(freqs_l1, splittings(map_omega, 1), c="tab:green", lw=1, alpha=0.5)
ax.plot(freqs_l2, splittings(map_omega, 2), c="tab:green", lw=2, alpha=0.5)
ax.plot(freqs_l3, splittings(map_omega, 3), c="tab:green", lw=3, alpha=0.5)


ax.plot(freqs_l1, splittings(true_omega, 1), c="tab:blue", lw=1, alpha=0.5)
ax.plot(freqs_l2, splittings(true_omega, 2), c="tab:blue", lw=2, alpha=0.5)
ax.plot(freqs_l3, splittings(true_omega, 3), c="tab:blue", lw=3, alpha=0.5)



raise a

# estimate autocorrelation
tau = sampler.get_autocorr_time()

tau_max = int(np.ceil(1.1 * np.max(tau)))

flat_samples = sampler.get_chain(
    discard=500, 
    thin=tau_max/2,
    flat=True
)

# discard first half.


fig = corner(
    samples[5000:].reshape((-1, ndim)),
    truths=true_params_vector
)
fig.savefig("casey_corner.png")


