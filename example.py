import corner
import emcee
import hierarchicalinc.HierarchicalAngles as HierarchicalAngles
import glob
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')
import os
import scipy
import scipy.interpolate as interpolate
import scipy.special as spec
import scipy.stats
import sys
import hierarchicalinc.utilities 

from tqdm import tqdm
from hierarchicalinc.models import model
from scipy import integrate
from scipy.integrate import quad
from hierarchicalinc.integrands import *

def import_data(star='ExampleLow', dir='ExampleRun/ExampleLow/', 
                root_dir='ExampleData/', npts=1000, plot=False, save=False):
    # Load in inclination angle samples
    
    fnames = glob.glob(root_dir+str(star)+'/samples*.txt')
    real = glob.glob(root_dir+str(star)+'/inc_angle.txt')
    real = np.loadtxt(real[0])
    # Load in and extract the input inclination angle for the simulation
    # npts - Number of points in posterior
    input_data = np.zeros([npts, len(fnames)])
    for idx, i in enumerate(fnames):
        samples = np.loadtxt(fnames[idx])
        input_data[:,idx] = np.random.choice(samples, npts, replace=False)

    # Fitting in cosine angle not in angle directly
    # Therefore prior is uniform in cosi
    input_data = np.cos(np.radians(input_data))
    if plot:
        # Always useful to visualise the data!
        for i in range(np.shape(input_data)[1]):
            plt.hist(input_data[:,i], bins=50, histtype='step', density=True)
        plt.xlim(0, 1)
        plt.axvline(np.cos(np.radians(real)), color='r', linestyle='--', label=r'Underlying Inclination Angle')
        plt.ylabel('Probability Density', fontsize=18)
        plt.xlabel(r'$\cos i_{\mathrm{s}}$', fontsize=18)
        plt.legend(loc='best')
        if save:
            plt.savefig(dir+'/'+str(star)+'/'+str(star)+'_data.png')
        #plt.close()
        plt.show()
    return input_data, real


def analyse_traces(sampler, data, prior, num, truth, plot=False, save=False):
    """
    Analyse traces
    """
    print(np.shape(sampler.chain))
    samples = sampler.chain.reshape(-1,np.shape(sampler.chain)[2])

    N = 10000
    xk = np.linspace(0, 200, N)
    yk = np.linspace(0.0, np.pi/2, N)

    # If want to sample from prior uncomment this
    #ps = prior.sample_from_prior(xk, yk, N)

    fig = corner.corner(samples,
                      labels=[r'$\kappa$', r'$\mu$'],
                  truths=[np.nan,
                          np.radians(truth)],
                  truth_color='#ff7f0e',
                  color='k',
                  alpha=0.01, normed=True);
    plt.show()

    km = np.percentile(sampler.flatchain[:,0], [16, 50, 84])
    mm = np.degrees(np.percentile(sampler.flatchain[:,1], [16, 50, 84]))
    print("UNDERLYING ANGLE: ", truth)
    print("HIERARCH ANGLE: ", mm[1])
    print("BOUNDS: ", mm[0], mm[2])

    n, bins, _ = plt.hist(np.degrees(sampler.flatchain[:,1]), bins=50, density=True,
             histtype='step', color='k', lw=2);
    plt.axvline(truth, linestyle='--', color='r')
    plt.hist(np.degrees(np.arccos(data)),
                          bins=50, density=True, histtype='step');
    plt.xlim(0, 90)
    plt.xlabel(r'Angle of Inclination (degrees)', fontsize=18)
    plt.ylabel(r'Probability Density', fontsize=18)
    plt.show()

    return mm

if __name__=="__main__":

    # Import data
    print("Importing data ...")
    star = str(sys.argv[1])
    directory = 'ExampleRun/'+str(star)+'/'
    if not os.path.exists(directory):
        print("Creating directory")
        os.makedirs(directory)
    else:
        print("Directory already exists")
    data, real_inc = import_data(star=star, dir=directory, npts=1000, plot=True, save=False)

    # Sampling
    print("... running sampler ...")
    sampler, prior = HierarchicalAngles.run_sampling(data, directory, star, nwalkers=20, niter=1000)
    # Analysis
    print("... analysing traces ...")
    params = analyse_traces(sampler, data, prior, star, real_inc)
