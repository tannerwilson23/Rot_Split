
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
import warnings

import numpy as np
import theano.tensor as tt

import pymc3 as pm
from pymc3.gp.cov import Covariance, Constant
from pymc3.gp.mean import Zero
from pymc3.gp.util import (conditioned_vars, infer_shape,
                           stabilize, cholesky, solve_lower, solve_upper)
from pymc3.distributions import draw_values
from theano.tensor.nlinalg import eigh


__all__ = ['Latent_Rot']



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

#freqs = np.concatenate((freqs_1,freqs_2,freqs_3))

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
#this needs to change of this to work.
#not sure if this means i have to change my freqs

#X2 = tt.sum(tt.square(flatx), 1)
#if Xs is None:
#    sqd = -2.0 * tt.dot(X, tt.transpose(X)) + (
#        tt.reshape(X2, (-1, 1)) + tt.reshape(X2, (1, -1))
#    )

x_diffs = np.zeros_like(flatx)
for j in range(1,len(flatx)):
    x_diffs[j] = flatx[j] - flatx[j-1]

x_diffs_eh = x_diffs[:, None]







#need to change here to be able to take in the range of the n values for the specific l
#need to spit out all of the possible frequencies and the splitting vlaues and these
#need to be the same size
def splittings(omega, l):
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
class RotSplitCov(pm.gp.cov.Covariance):
    
    def __init__(self, n, l):
        super(RotSplitCov, self).__init__(1, None)
        self.n = n
        self.l = l
        
    def square_dist(self, X, Xs):
        X = tt.mul(X, 1.0 / self.l)
        X2 = tt.sum(tt.square(X), 1)
        if Xs is None:
            sqd = -2.0 * tt.dot(X, tt.transpose(X)) + (
                tt.reshape(X2, (-1, 1)) + tt.reshape(X2, (1, -1))
            )
        else:
            Xs = tt.mul(Xs, 1.0 / self.l)
            Xs2 = tt.sum(tt.square(Xs), 1)
            sqd = -2.0 * tt.dot(X, tt.transpose(Xs)) + (
                tt.reshape(X2, (-1, 1)) + tt.reshape(Xs2, (1, -1))
            )
        return tt.clip(sqd, 0.0, np.inf)

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)

        prof = self.n**2 * tt.exp(-0.5 * self.square_dist(X, Xs)/self.l**2)
        return prof
#
#
with pm.Model() as model:

    ℓ = pm.Gamma("ℓ", alpha=2, beta=4)
    η = pm.HalfNormal("η", sd=1.0)

    cov_trend = RotSplitCov(η,ℓ)


    gp_trend = Latent_Rot(cov_func=cov_trend)
    f = gp_trend.prior("f", X = freqs_1)

    # The Gaussian process is a sum of these three components
    σ  = pm.HalfNormal("σ",  sd=10.0)

    y_ = pm.Normal('y', mu = f, sigma= σ, observed=split_vals_1)
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


class Base:
    R"""
    Base class.
    """

    def __init__(self, mean_func=Zero(), cov_func=Constant(0.0)):
        self.mean_func = mean_func
        self.cov_func = cov_func

    def __add__(self, other):
        same_attrs = set(self.__dict__.keys()) == set(other.__dict__.keys())
        if not isinstance(self, type(other)) or not same_attrs:
            raise TypeError("Cannot add different GP types")
        mean_total = self.mean_func + other.mean_func
        cov_total = self.cov_func + other.cov_func
        return self.__class__(mean_total, cov_total)

    def prior(self, name, X, *args, **kwargs):
        raise NotImplementedError

    def marginal_likelihood(self, name, X, *args, **kwargs):
        raise NotImplementedError

    def conditional(self, name, Xnew, *args, **kwargs):
        raise NotImplementedError

    def predict(self, Xnew, point=None, given=None, diag=False):
        raise NotImplementedError



@conditioned_vars(["X", "f"])
class Latent_Rot(Base):
    R"""
    Latent Gaussian process, converts the evaluated covariance into a mean function then applies rotational splittings.


    Parameters
    ----------
    cov_func : None, 2D array, or instance of Covariance
        The covariance function.  Defaults to zero.
    mean_func : None, instance of Mean
        The mean function.  Defaults to zero.

    Examples
    --------
    .. code:: python

        # A one dimensional column vector of inputs.
        X = np.linspace(0, 1, 10)[:, None]

        with pm.Model() as model:
            # Specify the covariance function.
            cov_func = pm.gp.cov.ExpQuad(1, ls=0.1)

            # Specify the GP.  The default mean function is `Zero`.
            gp = pm.gp.Latent(cov_func=cov_func)

            # Place a GP prior over the function f.
            f = gp.prior("f", X=X)

        ...

        # After fitting or sampling, specify the distribution
        # at new points with .conditional
        Xnew = np.linspace(-1, 2, 50)[:, None]

        with model:
            fcond = gp.conditional("fcond", Xnew=Xnew)
    """

    def __init__(self, mean_func=Zero(), cov_func=Constant(0.0)):
        super().__init__(mean_func, cov_func)

    def _build_prior(self, name, X, reparameterize=True, **kwargs):
        x = flatx
        mu = self.mean_func(x)
        cov = stabilize(self.cov_func(x))
        shape = infer_shape(X, kwargs.pop("shape", None))
        if reparameterize:
            v = pm.Normal(name + "_rotated_", mu=0.0, sigma=1.0, shape=shape, **kwargs)
            f = pm.Deterministic(name, mu + cholesky(cov).dot(v))
        else:
            f = pm.MvNormal(name, mu=mu, cov=cov, shape=shape, **kwargs)
        return f

    def prior(self, name, X, reparameterize=True, **kwargs):
        R"""
        Returns the GP prior distribution evaluated over the input
        locations `X`.

        This is the prior probability over the space
        of functions described by its mean and covariance function.

        .. math::

           f \mid X \sim \text{MvNormal}\left( \mu(X), k(X, X') \right)

        Parameters
        ----------
        name : string
            Name of the random variable
        X : array-like
            Function input values.
        reparameterize : bool
            Reparameterize the distribution by rotating the random
            variable by the Cholesky factor of the covariance matrix.
        **kwargs
            Extra keyword arguments that are passed to distribution constructor.
        """

        f = self._build_prior(name, X, reparameterize, **kwargs)
        self.X = X
        self.f = f
        #call splittings here
        vals = splittings(f,1)
        return vals

    def _get_given_vals(self, given):
        if given is None:
            given = {}
        if 'gp' in given:
            cov_total = given['gp'].cov_func
            mean_total = given['gp'].mean_func
        else:
            cov_total = self.cov_func
            mean_total = self.mean_func
        if all(val in given for val in ['X', 'f']):
            X, f = given['X'], given['f']
        else:
            X, f = self.X, self.f
        return X, f, cov_total, mean_total

    def _build_conditional(self, Xnew, X, f, cov_total, mean_total):
        Kxx = cov_total(X)
        Kxs = self.cov_func(X, Xnew)
        L = cholesky(stabilize(Kxx))
        A = solve_lower(L, Kxs)
        v = solve_lower(L, f - mean_total(X))
        mu = self.mean_func(Xnew) + tt.dot(tt.transpose(A), v)
        Kss = self.cov_func(Xnew)
        cov = Kss - tt.dot(tt.transpose(A), A)
        return mu, cov

    def conditional(self, name, Xnew, given=None, **kwargs):
        R"""
        Returns the conditional distribution evaluated over new input
        locations `Xnew`.

        Given a set of function values `f` that
        the GP prior was over, the conditional distribution over a
        set of new points, `f_*` is

        .. math::

           f_* \mid f, X, X_* \sim \mathcal{GP}\left(
               K(X_*, X) K(X, X)^{-1} f \,,
               K(X_*, X_*) - K(X_*, X) K(X, X)^{-1} K(X, X_*) \right)

        Parameters
        ----------
        name : string
            Name of the random variable
        Xnew : array-like
            Function input values.
        given : dict
            Can optionally take as key value pairs: `X`, `y`, `noise`,
            and `gp`.  See the section in the documentation on additive GP
            models in PyMC3 for more information.
        **kwargs
            Extra keyword arguments that are passed to `MvNormal` distribution
            constructor.
        """
        givens = self._get_given_vals(given)
        mu, cov = self._build_conditional(Xnew, *givens)
        shape = infer_shape(Xnew, kwargs.pop("shape", None))
        return pm.MvNormal(name, mu=mu, cov=cov, shape=shape, **kwargs)
