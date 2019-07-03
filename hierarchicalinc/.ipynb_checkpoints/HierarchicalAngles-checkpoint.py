import numpy as np
import scipy.stats
import scipy.interpolate as interpolate
import scipy.stats
import scipy

from scipy import integrate
from scipy.integrate import quad
from sini_inference_fns_mod import *
from utilities import inv_sample

class Likelihood:
    def __init__(self, _samples):
        """
        Proper docstrings
        """
        
        self.samples = _samples

    def __call__(self, params):

        mod = np.zeros(np.shape(self.samples))

        for i in range(np.shape(self.samples)[1]):
            mod[:,i] = model(self.samples[:,i], params)


        new_mod = mod.sum(axis=0) / float(np.shape(self.samples)[0])
        like = np.sum(np.log(new_mod))
        return like

class Prior(object):
    def __init__(self, _bounds):
        """
        Proper docstrings!
        """
        self.bounds = _bounds
        self.g = 50

    def sample_from_prior(self, x, y, N):
        # Prior on sigma
        pri_sigma = np.sin(y)
        # Prior on kappa
        pri = np.exp(-1.0 * np.log((np.pi * self.g) * (1 + (x / self.g)**2)))
        #pri = np.exp(np.log((1.0 + x**2.0)**(-3.0/4.0)))
        return np.c_[inv_sample(x, pri, N), inv_sample(y, pri_sigma, N)]

    def __call__(self, p):
        # We'll just put reasonable uniform priors on all the parameters.
        if not all(b[0] < v < b[1] for v, b in zip(p, self.bounds)):
            return -np.inf
        lnprior = 0.0
        lnprior += -1.0 * np.log((np.pi * self.g) * (1 + (p[0] / self.g)**2))
        # Isotropic prior placed on the location parameter mu        
        lnprior += np.log(np.sin(p[1]))
        return lnprior

class lnprob:
    def __init__(self, _prior, _like):
        self.prior = _prior
        self.like = _like

    def __call__(self, p):
        lp = self.prior(p)
        if not np.isfinite(lp):
            return -np.inf
        like = self.like(p)
        if not np.isfinite(like):
            return -np.inf
        return (lp + like)