
import os
import numpy as np
from scipy import (integrate, optimize as op)
from astropy import units as u
from astropy.table import Table
from glob import glob

import george
from george import kernels

import matplotlib.pyplot as plt

np.random.seed(0)


USE_GP = True

# Load betas.
def construct_beta_matrix(beta_table):
    ls = np.unique(beta_table["l"]).size
    ns = np.unique(beta_table["n"]).size

    beta = np.nan * np.ones((ls, ns))
    for row in beta_table:
        beta[row["l"] - 1, row["n"] - 1] = row["beta"]
    
    return beta

beta_table = Table.read("beta.dat", format="ascii")
beta = construct_beta_matrix(beta_table)


freq_table = Table.read("freq.dat", format="ascii")

omega_sampling = 100
l_degrees = (1, 2, 3) # TODO, this is hard coded elsewhere too

# TODO: why we do this?
n_lower, n_upper = (9, 27)
freq_nl = freq_table[(n_upper > freq_table["n"]) * (freq_table["n"] > n_lower)]

flat_frequencies = np.hstack(
    [freq_nl["Freqs"][freq_nl["l"] == l] for l in l_degrees]
)


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

r = np.loadtxt("x", skiprows=1)[::omega_sampling]

r_diffs = np.hstack([0, np.diff(r)])




def get_omega_arctan(params):
    a, b, c, d = params
    return a * np.arctan((r - 0.5 - c) * b) + d


mean_omega_init = 400

if not USE_GP:
        
    param_bounds = [
        (0, None),
        (-100, 100),
        (-0.5, 0.5),
        (0, None)
    ]

    get_omega = get_omega_arctan
    
    init_params = np.array([50, 90, -0.25, mean_omega_init])
    
else:


    kernel = 10 * kernels.RationalQuadraticKernel(log_alpha=1, metric=1)

    gp = george.GP(
        kernel,
        mean=mean_omega_init,
        fit_mean=True
    )
    gp.freeze_parameter("kernel:k2:metric:log_M_0_0")
    gp.compute(r)

    def get_omega(params):
        gp.set_parameter_vector(params)
        return gp.sample()

    init_params = gp.get_parameter_vector()

    param_bounds = [
        (0, 800),
        (-1, 5),
        (-6, 3),
    ]



def get_splittings(omega, l):
    area = np.dot(r_diffs, (omega * kern[l, n_lower + 1:n_upper, ::omega_sampling]).T)
    return area * beta[l - 1, n_lower:n_upper - 1]

def get_all_splittings(params, full_output=False, omega=None):
    if omega is None:
        omega = get_omega(params)
    splittings = np.hstack([get_splittings(omega, l) for l in l_degrees])
    return splittings if not full_output else (omega, splittings)



def ln_prior(params):
    if param_bounds is not None:
        for i, (lower_bound, upper_bound) in enumerate(param_bounds):
            if (lower_bound is not None and params[i] < lower_bound) \
            or (upper_bound is not None and params[i] > upper_bound):
                return -np.inf
        
    return 0

additive_variance = 0#15**2

def ln_likelihood(params, y, y_err, **kwargs):
    try:
        y_model = get_all_splittings(params, **kwargs)
    except:
        return -np.inf
    
    ivar = 1.0/(y_err**2 + additive_variance)
    return -0.5 * np.sum((y - y_model)**2 * ivar)

def ln_probability(params, y, y_err, verbose=False):
    lp = ln_prior(params)
    if not np.isfinite(lp):
        if verbose: print(params, -np.inf)
        return -np.inf
    
    lp += ln_likelihood(params, y, y_err)
    if verbose: print(params, lp)
    return lp


true_params = [20, 70, -0.2, 440]

x = flat_frequencies
true_omega = get_omega_arctan(true_params)
true_splittings = get_all_splittings(true_params, omega=true_omega)
N = x.size

y_err = 0.1 * np.random.randn(N)
y = true_splittings + y_err * np.random.randn(N)

args = (y, y_err)

print("Target: {}".format(ln_likelihood(true_params, *args, omega=true_omega)))

"""
def get_splittings2(omega, l, ts):
    K = kern[l, n_lower + 1:n_upper, ::omega_sampling]
    R = r_diffs
    O = omega
    B = beta[l - 1, n_lower:n_upper - 1]

    area = np.dot(r_diffs, (omega * K).T)
    foo = area * beta[l - 1, n_lower:n_upper - 1]

    # ts = R @ (O * K).T * B
    # (O * K) @ R.T = (ts / B).T
    O2 = np.array([np.ones(17) * o for o in O]).T
    
    foo2 = np.atleast_2d(ts / B).T
    # (O * K) @ R.T = foo2

    def objective_function(O):
        return (R @ (O*K).T * B - ts).flatten()

    #p_opt = op.leastsq(objective_function, 400 * np.ones_like(O))
    U, S, V = np.linalg.svd(O2)


    raise a

get_splittings2(true_omega, 1, true_splittings[:17])

area_invert = true_splittings[:17]/beta[0, n_lower:n_upper - 1]
# area_invert = r_diffs @ (O @ K).T


assert False
"""


# Draw true things.
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].plot(
    r, 
    true_omega, 
    c="tab:blue"
)
axes[0].set_xlabel(r"$r/R$")
axes[0].set_ylabel(r"$\Omega$")

axes[1].scatter(
    x,
    y,
    c="tab:blue"
)
axes[1].errorbar(
    x,
    y,
    yerr=y_err,
    fmt="none",
    c="tab:blue"
)
axes[1].set_xlabel("frequency")
axes[1].set_ylabel("splitting")
fig.tight_layout()


# Now try one.
init_omega, init_splittings = get_all_splittings(init_params, full_output=True)

axes[0].plot(
    r,
    init_omega,
    c="tab:red"
)
axes[1].scatter(
    x, 
    init_splittings,
    c="tab:red"
)

# Be verbose when optimizing, quiet otherwise.
nlp = lambda *args, **kwds: -ln_probability(*args, verbose=True, **kwds)

# Let's optimize.
p_opt = op.minimize(
    nlp,
    init_params,
    args,
    method="Nelder-Mead",
    options=dict(
        maxiter=1000,
        maxfev=1000
    ),
    bounds=param_bounds
)
print(p_opt)


opt_omega, opt_splittings = get_all_splittings(p_opt.x, full_output=True)
axes[0].plot(
    r,
    opt_omega,
    c="tab:green"
)
axes[1].scatter(
    x,
    opt_splittings,
    c="tab:green"
)


import emcee

nwalkers = 32
ndim = p_opt.x.size
nsamples = 20000

sampler = emcee.EnsembleSampler(
    nwalkers,
    ndim,
    ln_probability,
    args=args
)

p0 = p_opt.x + 1e-3 * np.random.randn(nwalkers, ndim)

state = sampler.run_mcmc(p0, nsamples, progress=True)

samples = sampler.get_chain()

fig, axes = plt.subplots(ndim)
for i in range(ndim):
    axes[i].plot(
        samples[:, :, i],
        c="k",
        alpha=0.3
    )

from corner import corner

flat_samples = samples[int(nsamples/2):].reshape((-1, ndim))
fig = corner(flat_samples)


# Draw true things.
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].plot(
    r, 
    true_omega, 
    c="tab:blue"
)
axes[0].set_xlabel(r"$r/R$")
axes[0].set_ylabel(r"$\Omega$")

axes[1].scatter(
    x,
    y,
    c="tab:blue"
)
axes[1].errorbar(
    x,
    y,
    yerr=y_err,
    fmt="none",
    c="tab:blue"
)
axes[1].set_xlabel("frequency")
axes[1].set_ylabel("splitting")
fig.tight_layout()


for idx in np.random.choice(flat_samples.shape[0], 100):
    s_omega, s_splittings = get_all_splittings(
        flat_samples[idx], 
        full_output=True
    )

    axes[0].plot(
        r,
        s_omega,
        c="#666666",
        alpha=0.2,
        zorder=-1
    )
    axes[1].scatter(
        x,
        s_splittings,
        c="#666666",
        alpha=0.2,
        s=1,
        zorder=-1
    )