import numpy as np

from scipy import integrate
from scipy.integrate import quad
from sini_inference_fns_mod import *

def cosine_fisher_pdf(y, k, mu):
    """
    Proper docstring!
    """
    if k == 0:
        norm = quad(cosine_fisher_integrand, 0, 1.0, args=(k, mu))[0]
        return np.ones(len(y)) / norm
    elif mu == 0:
        norm = quad(cosine_fisher_integrand, 0, 1.0, args=(k, mu))[0]
        return k / np.sinh(k) * np.exp(k*y) / norm
    else:
        norm = quad(cosine_fisher_integrand, 0, 1.0, args=(k, mu))[0]
        print(norm)
        return k / np.sinh(k) * np.exp(k*(y*np.cos(mu) + np.sqrt(1-y**2)*np.sin(mu))) / norm

def model(y, p):
    """
    Cosine of Fisher distribution with location
    """
    kappa, mu = p
    if kappa == 0.0:
        mod = np.ones(len(y))
    else:
        mod = kappa / np.sinh(kappa) * np.exp(kappa*(y*np.cos(mu) + np.sqrt(1-y**2)*np.sin(mu)))
    norm = quad(cosine_fisher_integrand, 0, 1.0, args=(kappa, mu))[0]
    return mod / norm