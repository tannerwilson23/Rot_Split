import sys
import pymc3 as pm
import theano.tensor as tt
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
import os
from scipy.integrate import simps,cumtrapz

#reading in data

x_small = pd.read_table('x')
x_small = np.array(x_small)
flatx = x_small.flatten()

n = 4800

kernels = []
for filename in os.listdir('kerns'):
    k = pd.read_table(os.path.join('kerns', filename))
    kernels += [k]

K_i = pd.concat(kernels,axis = 1, sort= False)


beta = pd.read_table('beta.dat', sep='\s+')
np.array(beta.loc[beta['l'] == 1]['beta'])

frequ = pd.read_table('fgong-freqs.dat', sep='\s+')

#splittings for the initial plot, can use simps here

def splittings_basic(omega, x, lval):
    for l in [lval]:
        vals = []
        freqs = np.array(frequ.loc[frequ['l'] == l]['nu'])
        for i in range(1,34):

            ker = np.array(K_i['l.'+str(l)+'_n.'+str(i)])
            bet = np.array(beta.loc[beta['l'] == 1]['beta'])[i]
            if (np.shape(x)[0] < 4800):
                ker = ker[0::148]
            for m in [1]:
                delt = m * bet * simps(omega*ker, x)
                vals.append(delt)
    np.array(freqs)
    return freqs, vals


# Define the true covariance function and its parameters
ℓ_true = 0.5
η_true = 0.01
cov_func = η_true**2 * pm.gp.cov.ExpQuad(1, ℓ_true)

# A mean function that is zero everywhere
a_true = 0.4
b_true = 10
#b_true = (b_true/2)**(1/2)
c_true = 0.4
mean_func = a_true * np.exp(-x_small*b_true) + c_true
mean_func = mean_func.flatten()


#true rotation profile with covariance as defined above
f_true = np.random.multivariate_normal(mean_func,
                                       cov_func(x_small).eval() + 1e-8*np.eye(n), 1).flatten()

#find the frequencies and the splitting values
freqs,vals = splittings_basic(mean_func, flatx, 1)


#add some noise to the observed splittings
σ_true = 0.0004
y = vals + σ_true * np.random.randn(np.shape(vals)[0])


plt.plot(freqs[0:-1],vals, label = 'True Splittings')
plt.plot(freqs[0:-1],y, label = 'Observed Splittings')
plt.legend()
plt.xlabel('Frequency')
plt.ylabel('Splitting Value')


#splittings using theano.tensor
def splittings(omega, x1, lval):
    print(x1)
    vals = []
    for l in [lval]:
        freqs = np.array(frequ.loc[frequ['l'] == l]['nu'])
        for i in range(1,34):
            area = 0
            ker = np.array(K_i['l.'+str(l)+'_n.'+str(i)])
            bet = np.array(beta.loc[beta['l'] == 1]['beta'])[i]
            if (np.shape(x1)[0] < 4800):
                ker = ker[0::148]
            for j in range(len(x1)):
                print(i,j)
                area_curr = (x1[j]-x1[j-1])*tt.dot(omega[j],ker[j])
                area = tt.add(area,area_curr)
            delt1 = tt.dot(bet,area)
            vals.append(delt1)
    vals = tt.as_tensor_variable(vals)
#    vals = tt.flatten(vals)
    vals = tt.squeeze(vals)
    print(vals.ndim)
    return freqs, vals


#gp mean function including splittings
class CustomMeanSplittings(pm.gp.mean.Mean):

    def __init__(self, a = 0, b = 1, c = 0):
        pm.gp.mean.Mean.__init__(self)
        self.a = a
        self.b = b
        self.c = c


    def __call__(self, big_X):
        print(len(big_X))
        rot_prof = tt.squeeze(self.a*tt.exp(tt.dot(-x_small,self.b))+self.c)
        rot_prof
        print(rot_prof)
        freqs,vals = splittings(rot_prof, x_small, 1)
        return vals


# A one dimensional column vector of inputs.


#our gp model
with pm.Model() as model:

    ℓ = pm.Gamma("ℓ", alpha=2, beta=4)
    η = pm.HalfNormal("η", sigma=2.0)



    cov_trend = η**2 * pm.gp.cov.ExpQuad(1, ℓ)


    a_var = pm.Uniform("a_var", lower= 0.3, upper= 0.5)
    b_var = pm.Uniform("b_var", lower=0.1, upper = 0.3)
    c_var = pm.HalfNormal("c_var", sigma=0.5)


    mean_trend = CustomMeanSplittings(a = a_var, b= b_var, c= c_var)


    #cov_big = tt.identity_like(mean_trend(x_var_flat))*cov_trend

    gp_trend = pm.gp.Latent(mean_func = mean_trend , cov_func=cov_trend)
    #mean_trend(x_var_flat).eval()

    f = gp_trend.prior("f", X = freqs)

    # The Gaussian process is a sum of these three components
    σ  = pm.HalfNormal("σ",  sigma=1.0)

    y_ = pm.Normal('y', mu = f, sigma = σ, observed = y)

    # this line calls an optimizer to find the MAP
    #mp = pm.find_MAP(include_transformed=True)


    trace = pm.sample(1000, chains=1, tune=1000)
