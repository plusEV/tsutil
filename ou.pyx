cimport cython
import numpy as np
cimport numpy as np


def lnprior(theta):
    mu, sig, lam = theta
    if -1 < mu < 1 and 0.001 < sig < 1.0 and 0.001 < lam < 3.0:
        return 0.0
    return -np.inf

def lnlike(theta, x, y):
    mu, sig, lam = theta
    n = len(x)
    sighat =np.sqrt( sig**2*((1-np.exp(-2*lam))/(2*lam)))
    t1 = -n/2.*np.log(2*np.pi)
    t2 = -n*np.log(sighat)
    model = x * np.exp(-1*lam) + mu * (1. - np.exp(-1*lam))
    err = np.sum((y - model)**2)
    t3 = -1/(2*sighat**2) * err
    return t1 + t2 + t3

def lnprob(theta, x, y):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y)

def OU_MLE(price_array,delta=1,which=None):
    n = float(len(price_array) - 1)
    x = np.sum(price_array[:-1])
    y = np.sum(price_array[1:])
    xx = np.sum(price_array[:-1]**2)
    yy = np.sum(price_array[1:]**2)
    xy = np.sum(price_array[:-1]*price_array[1:])
    
    
    mu = (y*xx - x*xy) / ( n * (xx - xy) - (x**2 - x*y) ) 
    lam = -1 * np.log(( xy - mu*x - mu*y + n*mu**2) / (xx - 2*mu*x + n*mu**2 )) / delta
    a = np.exp(-1*lam*delta)
    sigmah2 = (yy - 2*a*xy + a**2*xx - 2*mu*(1-a)*(y - a*x) + n*mu**2*(1-a)**2) / n
    sigma = np.sqrt(sigmah2*2*lam/(1-a**2))
    if which:
        if which == "mu":
            return mu
        if which == "sigma":
            return sigma
        if which == "lam":
            return lam
    return [mu,sigma,lam]

def cy_generate_OU(double mu,double sigma,double lam,long nsteps,double dt,double init):
    cdef:
        np.ndarray[double, ndim=1] S = np.zeros(nsteps, dtype=np.double)
        np.ndarray[double, ndim=1] dWt
        long n = nsteps+1,t=0
        
    S[0] = init
    np.random.seed(42)
    dWt = np.random.normal(0,size=n)
    
    if lam!=0:
        dWt*= np.sqrt((1-np.exp(-2*lam*dt))/(2*lam))
    
    lamt = np.exp(-1*lam*dt)
    for 1 <= t < (n-1):
        S[t] = S[t-1] * lamt + mu*(1-lamt) + sigma*dWt[t]
        
    return S  
