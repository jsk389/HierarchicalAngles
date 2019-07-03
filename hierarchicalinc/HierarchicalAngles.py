import emcee
import numpy as np
import scipy.stats
import scipy.interpolate as interpolate
import scipy.stats
import scipy
from . import utilities

from .integrands import *
from .models import model
from scipy import integrate
from scipy.integrate import quad
from tqdm import tqdm

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
        return np.c_[utilities.inv_sample(x, pri, N), utilities.inv_sample(y, pri_sigma, N)]

    def __call__(self, p):
        # We'll just put reasonable uniform priors on all the parameters.
        if not all(b[0] < v < b[1] for v, b in zip(p, self.bounds)):
            return -np.inf
        lnprior = 0.0
        # Cauchy prior on the concentration parameter kappa
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

def run_sampling(data, save_dir, fname, burnin=0, niter=2000, nwalkers=200, plot=False, threads=4):
    """
    Run sampling
    """
    # Set up prior bounds
    bounds = [(0.0, 200.0), (0.0, np.pi/2.0)]

    # Initial guesses
    params = [100.0, np.arccos(np.mean(data.ravel()))]

    # Initialise prior, likelihood and logprob functions
    prior = Prior(bounds)
    like = Likelihood(data)
    logprob = lnprob(prior, like)

    # Run MCMC
    nwalkers, niter, ndims = nwalkers, niter, int(len(params))
    p0 = [params + 1.0e-2*np.random.randn(ndims) for k in range(nwalkers)]
    
    # Name for saved file
    backend_fname = str(save_dir)+str(fname)+'-chains.h5'
    backend = emcee.backends.HDFBackend(backend_fname)
    
    # Clear file incase it already exists
    backend.reset(nwalkers, ndims)
    sampler = emcee.EnsembleSampler(nwalkers, ndims, logprob, threads=threads, backend=backend)
    
    print("Running first burn-in...")
    for p0, lp, _ in tqdm(sampler.sample(p0, iterations=niter), total=niter):
        pass
    sampler.reset()

    print("Running production...")
    for p, lp, _ in tqdm(sampler.sample(p0, iterations=2000), total=2000):
        pass

    return sampler, prior