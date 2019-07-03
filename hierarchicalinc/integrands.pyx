import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float
ctypedef np.float_t DTYPE_t
cdef DTYPE_t pi = 3.1415926535897932384626433832795

cdef extern from "math.h":
    double sin(double)
    double cos(double)
    double tanh(double)
    double sqrt(double)
    double atan2(double,double)
    double acos(double)
    double asin(double)
    double abs(double)
    double log(double)
    double ceil(double)
    double cosh(double)
    double sinh(double)
    double exp(double)

def gaussian_integrand(DTYPE_t y, DTYPE_t mu, DTYPE_t sigma):
    """ Integrand to compute normalisation constant for angle distributed as Gaussian
        in cosine space
    """
    return (1.0 / sqrt(1.0 - y*y)) * (1.0 / sqrt(2.0 * pi * sigma*sigma)) * exp(-((acos(y)-mu)*(acos(y)-mu))/(2.0*sigma*sigma))

def fisher_integrand(DTYPE_t theta, DTYPE_t k, DTYPE_t mu):
    """ Integrand to compute normalisation constant for Fisher with location
    """
    if k == 0:
        return sin(theta)
    elif mu == 0:
        return k / sinh(k) * exp(k*cos(theta)) * sin(theta)
    else:
        return k / sinh(k) * exp(k*cos(theta - mu)) * sin(theta)

def sine_fisher_integrand(DTYPE_t y, DTYPE_t k, DTYPE_t mu):
    """ Integrand to compute integrand in sine of Fisher with location
    """
    if k == 0.0:
        return 2.0 * y / sqrt(1 - y*y)
    elif mu == 0.0:
        return y / (sqrt(1 - y*y) * cosh(k * sqrt(1 - y*y)))
    else:
        return y / sqrt(1 - y*y) * (exp(k*(sqrt(1 - y*y)*cos(mu) + y*sin(mu))) + exp(k*(-1.0*sqrt(1 - y*y)*cos(mu) + y*sin(mu))))

def cosine_fisher_integrand(DTYPE_t y, DTYPE_t k, DTYPE_t mu):
    """ Integrand to compute integrand in sine of Fisher with location
    """
    if k == 0.0:
        return 1.0
    if mu == 0.0:

        return k / sinh(k) * exp(k*y)
    elif mu == np.pi/2.0:

        return k / sinh(k) * exp(k*sqrt(1 - y*y))
    else:
        return k / sinh(k) * exp(k*(y*cos(mu) + sqrt(1 - y*y)*sin(mu)))


def cosine_integrand(DTYPE_t y, DTYPE_t k, DTYPE_t mu, DTYPE_t z):
    """ Integrand to compute integrand in cosine i with Fisher including location
    """
    if k == 0.0:
        return 2.0 / sqrt(1 - y*y) / sqrt(1-(z/y)*(z/y))
    elif mu == 0.0:
        return cosh(k*sqrt(1-y*y)) / sqrt(1-y*y) / sqrt(1-(z/y)*(z/y))
    elif mu == np.pi/2.0:
        return exp(k * y * sin(mu)) / sqrt(1-y*y) / sqrt(1-(z/y)*(z/y))
    else:
        return (exp(k*(sqrt(1-y*y)*cos(mu) + y*sin(mu))) + exp(k*(-1.0*sqrt(1-y*y)*cos(mu) + y*sin(mu)))) / sqrt(1 - y*y) / sqrt(1 - (z/y)*(z/y))

