import glob
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Math, display
import scipy.special as spec
import scipy.stats
import pystan
import os
import scipy.interpolate as interpolate
import scipy.stats
from pyqt_fit import kde, kde_methods
import scipy
from tqdm import *
import emcee
import corner
from scipy import integrate
from scipy.integrate import quad
from sini_inference_fns_mod import *
import sys


def import_data(num, npts=1000, plot=False):
    # Load in inclination angle samples
    root_dir = home+'/Dropbox/Python/Peak-Bagging/Red_Giants/'
    fnames = glob.glob(root_dir+str(num)+'/*/samples.txt')
    # Load in parameter files for each mode
    lim_params = glob.glob(root_dir+str(num)+'/*/limit_parameters.txt')
    # Load in and extract the input inclination angle for the simulation
    idx = 0
    # npts - Number of points in posterior
    input_data = np.zeros(npts)
    for idx, i in enumerate(fnames):
        tmp_m, tmp_p, tmp_n = np.loadtxt(lim_params[idx], unpack=True)
        #if (calc_hb(tmp_m[2], tmp_m[3], tmp_m[0]) > 18.9) & (calc_red_split(tmp_m[4], tmp_m[3]) > 1e-2):
        # Make cuts in H/B and linewidth
        # 18/04/2018 DON'T MAKE CUTS!        
        #	if (calc_hb(tmp_m[2], tmp_m[3], tmp_m[0]) > 18.9) & (tmp_m[3] > 1.0*0.007):
        samples = np.loadtxt(fnames[idx])
        try:
            input_data = np.c_[input_data, np.random.choice(samples, npts, replace=False)]
        except:
            input_data = np.c_[input_data, np.random.choice(samples[:,-1], npts, replace=False)]
    input_data = input_data[:,1:]
    # Fitting in cosine angle not in angle directly
    # Therefore prior is uniform in cosi
    input_data = np.cos(np.radians(input_data))
    #if plot:
        # Always useful to visualise the data!
    for i in range(np.shape(input_data)[1]):
        plt.hist(input_data[:,i], bins=50, histtype='step', normed=True)
    plt.xlim(0, 1)
    plt.ylabel('Probability Density')
    plt.xlabel(r'$\cos i_{\mathrm{s}}$')
    plt.savefig(str(num)+'/'+str(num)+'_data.png')
    plt.close()
    return input_data

def compute_mean_of_samples(samples):
    """
    Compute distribution of means from posterior samples
    """
    means = np.mean(samples, axis=1)
    np.savetxt(str(num)+'/'+str(num)+'_means.txt', means)

def run_sampling(data, num, burnin=0, plot=False):
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

    # Firstly compute distribution of means non-parametrically
    compute_mean_of_samples(np.degrees(np.arccos(data)))
    # Run MCMC
    nwalkers, niter, ndims = 200, 2000, int(len(params))
    #param_names = [r'$\kappa$', r'$\mu$']
    p0 = [params + 1.0e-2*np.random.randn(ndims) for k in range(nwalkers)]

    backend_fname = str(num)+'/'+str(num)+'_chains.h5'
    backend = emcee.backends.HDFBackend(backend_fname)
    # Clear file incase it already exists
    backend.reset(nwalkers, ndims)
    sampler = emcee.EnsembleSampler(nwalkers, ndims, logprob, threads=4, backend=backend)
    
    print("Running first burn-in...")
    for p0, lp, _ in tqdm(sampler.sample(p0, iterations=2000), total=2000):
        pass

    sampler.reset()
    print("Running production...")
    #sampler.run_mcmc(p0, 1000);
    for p, lp, _ in tqdm(sampler.sample(p0, iterations=2000), total=2000):
        pass

    # Compute effective sample size
    tau, neff = compute_effective_sample_size(sampler.flatchain)
    print("Effective sample size: {} out of {} total samples".format(neff, len(sampler.flatchain)))
    print("Autocorrelation length: {}".format(tau))

    return sampler, prior, tau

def analyse_traces(sampler, data, prior, tau, num, plot=False):
    """
    Analyse traces
    """

    pmax = sampler.chain.reshape(-1,2)[np.argmax(sampler.lnprobability.reshape(-1, 1))]

    N = 10000
    xk = np.linspace(0, 200, N)
    yk = np.linspace(0.0, np.pi/2, N)

    ps = prior.sample_from_prior(xk, yk, N)

    fig = corner.corner(ps,
                      labels=[r'$\kappa$', r'$\mu$'],
                  truths=[np.median(sampler.flatchain[::int(tau)], axis=0)[0],
                          np.median(sampler.flatchain[::int(tau)], axis=0)[1]],
                  truth_color='#ff7f0e',
                  color='#d62728',
                  alpha=0.01, normed=True);
    fig = corner.corner(sampler.flatchain[::int(tau)],
                      labels=[r'$\kappa$', r'$\mu$'],
                      truths=[pmax[0], pmax[1]],
                      truth_color='#1f77b4',
                      fig=fig, normed=True);
    plt.savefig(str(num)+'/'+str(num)+'_corner.png')
    plt.close()
    med = [np.median(sampler.flatchain[::int(tau)], axis=0)[0],
            np.median(sampler.flatchain[::int(tau)], axis=0)[1]]
    x = np.linspace(0, 1, N)
    #np.median(sampler.flatchain[::int(tau)], axis=0))
    r = inv_sample(x, model(x, med), N)
    #np.savetxt(str(num)+'/'+str(num)+'_posts.txt', r)
    h_vals = compute_values(np.degrees(np.arccos(r)), 0.683)
    km = compute_values(sampler.flatchain[::int(tau),0], 0.683)
    mm = compute_values(sampler.flatchain[::int(tau),1], 0.683)
    print("HIERARCH: ", h_vals[0])
    # Save out summary statistics (in i_s not cosine i_s)
    np.savetxt(str(num)+'/'+str(num)+'_params.txt', np.r_[km, mm])
    #np.savetxt(str(num)+'/'+str(num)+'')
    np.savetxt(str(num)+'/'+str(num)+'_values.txt', h_vals)
    # Save out summary statistics (in i_s not cosine i_s)

    n, bins, _ = plt.hist(np.degrees(np.arccos(r)), bins=100, normed=True,
             histtype='step', color='r', linestyle='--', lw=2);
    plt.hist(np.degrees(np.arccos(data)),
                          bins=100, normed=True, histtype='step');
    bin_centres = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bin_centres, n)
    plt.xlim(0, 90)
    plt.xlabel(r'Angle of Inclination (degrees)', fontsize=18)
    plt.ylabel(r'Probability Density', fontsize=18)
    plt.savefig(str(num)+'/'+str(num)+'_inc_check.png')
    plt.close()

    return pmax #np.median(sampler.flatchain[::int(tau)], axis=0)


def combine_inference(data, params, num):
    """
    Combine inference and posteriors
    """

    k, m = params
    x = np.linspace(0, 1, 10000)
    bets = np.zeros([10000, np.shape(data)[1]])
    angles_real = np.zeros_like(bets)

    for j in range(np.shape(data)[1]): # this is the real bit that makes the histograms

        bets[:,j] = model(x, [k, m])
        est_large = kde.KDE1D(data[:,j], lower=0, upper=1,
                              method=kde_methods.reflection)
        angles_real[:,j] = est_large(x)
    # Normalise
    bets /= np.nansum(bets, axis=0)
    angles_real /= np.nansum(angles_real, axis=0)

    # Combine
    y = angles_real * bets
    norm = np.nansum(y, axis=0)
    # Normalise
    y /= norm
    # Plot
    plt.plot(x, y)
    plt.plot(x, angles_real, ':')
    #print(np.prod(angles_real, axis=1) / np.sum(np.prod(angles_real, axis=1)))
    plt.plot(x,  bets[:,0], lw=2, c='k')#np.prod(angles_real, axis=1) / np.sum(np.prod(angles_real, axis=1)))
    plt.xlim(0, 1)
    plt.xlabel(r'$\cos i_{\mathrm{s}}$', fontsize=18)
    plt.ylabel(r'Probability Density', fontsize=18)
    plt.savefig(str(num)+'/'+str(num)+'_cosine_check.png')
    plt.close()

    # Take product of posteriors
    tt = np.prod(y, axis=1)
    norm = np.sum(tt)
    # Compare against simple multiplying posteriors together
    old = np.prod(angles_real, axis=1) / np.sum(np.prod(angles_real, axis=1))


    print("HIERARCH: ", np.degrees(np.arccos(x[np.argmax(tt/norm)])))
    # Inverse Transform sample for plot
    new_dat = inv_sample(x, tt/norm, len(x))
    old_dat = inv_sample(x, old, len(x))
    plt.plot(np.degrees(np.arccos(x)), tt/norm, label=r'KDE')
    plt.hist(np.degrees(np.arccos(new_dat)), bins=100, normed=True, histtype='step', label=r'New')
    plt.hist(np.degrees(np.arccos(old_dat)), bins=100, normed=True, histtype='step', label=r'Simple')

    plt.xlabel(r'Inclination Angle (degrees)', fontsize=18)
    plt.ylabel(r'Probability Density', fontsize=18)
    plt.legend(loc='best')
    plt.savefig(str(num)+'/'+str(num)+'_final_check.png')
    plt.close()

    h_vals = hpd(np.degrees(np.arccos(new_dat.ravel())), 0.683)
    old_vals = hpd(np.degrees(np.arccos(old_dat)), 0.683)

    # Save out model and samples
    np.savetxt(str(num)+'/'+str(num)+'_hierarch_dist_pmax.txt', np.c_[x, tt/norm, new_dat])
    np.savetxt(str(num)+'/'+str(num)+'_non_hierarch_dist.txt', np.c_[x, old, old_dat])

    # Save out summary statistics (in i_s not cosine i_s)
    np.savetxt(str(num)+'/'+str(num)+'_values_pmax.txt',
               np.r_[h_vals, old_vals])
    return tt

if __name__=="__main__":


    list_dirs = np.loadtxt(sys.argv[1], usecols=(0,), dtype='float').astype(int)

    #list_dirs = list_dirs[:40]
    for idx in range(len(list_dirs)):
        num = list_dirs[idx]
        print("Star {} of {}".format(idx+1, len(list_dirs)))
        # Import data
        print("Importing data ...")
        directory = str(num)
        if not os.path.exists(directory):
            print("Creating directory")
            os.makedirs(directory)
        else:
            print("Directory already exists")
        try:
            data = import_data(num, npts=1000)
            # Create folder if doesn't exist
            if np.shape(data)[1] > 1:
                # Sampling
                print("... running sampler ...")
                trace, prior, tau = run_sampling(data, num)
                # Analysis
                print("... analysing traces ...")
                params = analyse_traces(trace, data, prior, tau, num)
                # Combine
                #print("... combining inference & final plotting ...")
                #hierarch_dist = combine_inference(data, params, real_inc, num)
                # Final plot

            else:
                print("Only one mode present - skipping!")
        except:
            print("Skipping!")
