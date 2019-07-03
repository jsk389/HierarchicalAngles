import numpy as np
import scipy.interpolate as interpolate

# Basic function for inverse transform sampling
def inv_sample(vals, pdf, N):
    cdf = pdf.cumsum()
    cdf /= cdf.max()
    inv_cdf = interpolate.interp1d(cdf, vals)
    # Don't start from zero to avoid out of bounds error in
    # interpolation
    r = np.random.uniform(cdf.min()+0.0001, 1.0, N)
    return inv_cdf(r)