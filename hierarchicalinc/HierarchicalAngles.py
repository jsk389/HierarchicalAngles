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
        Likelihood function as defined by equation (12) of Kuszlewicz et al. (2019)
        
        Args:
        _samples (array): A numpy array containing the posterior samples
        """
        
        self.samples = _samples

    def __call__(self, params):
        """
        Args:
        params (array): Parameters of the model (concentration parameter kappa and 
                        location parameter mu)
        """
        # Set up array containing likelihood values
        mod = np.zeros(np.shape(self.samples))
        
        # Compute likelihood
        for i in range(np.shape(self.samples)[1]):
            mod[:,i] = model(self.samples[:,i], params)

        # Compute sum
        new_mod = mod.sum(axis=0) / float(np.shape(self.samples)[0])
        # Compute product in log-space to give log-likelihood
        like = np.sum(np.log(new_mod))
        return like

class Prior(object):
    def __init__(self, _bounds):
        """
        Prior distributions used in the inference
        
        Args:
        _bounds (list): list of tuples containing the upper and lower bounds of the uniform
                        priors.
        """
        self.bounds = _bounds
        # Width of the Half-Cauchy distribution
        self.g = 50

    def sample_from_prior(self, x, y, N):
        """
        Function to enable sampling from the prior
        
        Args:
        x (array): array containing kappa values at which to sample prior
        y (array): array containing my values at which to sample prior
        N (int): Number of samples to draw - must be same size as x and y
        """
        # Prior on sigma
        pri_sigma = np.sin(y)
        # Prior on kappa
        pri = np.exp(-1.0 * np.log((np.pi * self.g) * (1 + (x / self.g)**2)))
        return np.c_[utilities.inv_sample(x, pri, N), utilities.inv_sample(y, pri_sigma, N)]

    def __call__(self, p):
        """
        Args:
        p (array): array containing parameters at which to evaluate prior
        """
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
    """
    Log-probability function
    
    Args:
    _prior (class): Prior distribution class
    _like (class): Likelihood function class
    """
    def __init__(self, _prior, _like):
        self.prior = _prior
        self.like = _like

    def __call__(self, p):
        """
        Args:
        p (array): array containing parameters
        """
        # Evaluate prior
        lp = self.prior(p)
        # If prior probability not finite then return negative infinity, i.e. probability of zero
        if not np.isfinite(lp):
            return -np.inf
        # Evaluation likelihood
        like = self.like(p)
        # Same check for likelihood
        if not np.isfinite(like):
            return -np.inf
        return (lp + like)

def run_sampling(data, save_dir, fname, burnin=0, niter=2000, nwalkers=200, plot=False, threads=4):
    """
    Run sampling
    
    Args:
    data (array): array containing inclination angle posterior samples
    save_dir (str): directory to save chains
    fname (str):
    burnin (int): number of steps to use as burnin
    niter (int): number of iterations for burnin and production MCMC run, default=2000.
    nwalkers (int): number of walkers to use in MCMC, default=200.
    plot (bool):
    threads (int): number of threads for MCMC, default=4.
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
