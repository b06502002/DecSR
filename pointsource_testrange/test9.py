import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
np.random.seed(1234321)
## This file mainly checks 
    # 1. The intensity
    # 2. 

def uniform_proposal(x, y=0, delta=2.0):
    return np.random.uniform([x-delta,y-delta], [x+delta, y+delta],size=(1,2))

def metropolis_sampler(p, nsamples, proposal=uniform_proposal):
    x, y = 3, 3
    for i in range(nsamples):
        trial = proposal(x, y)
        acceptance = p(trial[0][0], trial[0][1])/p(x, y)
        if np.random.uniform() < acceptance:
            x = trial[0][0]
            y = trial[0][1]
        yield x, y

def setAlpha(beta, r50):
    """
    This function set the alpha value with the given relationship. According to the paper (Fast PSF modeling with DL),
    the half light radius is set as 1 pixel."""
    return r50/np.sqrt(2**(1/(beta-1))-1)

def Moffat(x, y, alpha, beta, I0):
    r = np.sqrt(x**2+y**2)
    return I0*(1+(r/alpha)**2)**-beta # 2*(beta-1)/alpha**2*(1+(r/alpha)**2)**-beta

def size_module(beta, f):
    return f*(np.sqrt(2**(1/(beta-1))-1)/(2*np.sqrt(2**(1/beta)-1)))

# basic parameter section
b = 3.419 # beta in the base profile
FWHM = 4
R50 = size_module(b, FWHM)
p = lambda x, y: Moffat(x, y, setAlpha(b, R50), b, 1)
samples = list(metropolis_sampler(p, 100000))
samples[0] = tuple(elem*R50 for elem in samples[0])
samples[1] = tuple(elem*R50 for elem in samples[1])

unzipped_object = zip(*samples)
unzipped_sample = list(unzipped_object)

w = np.ones(100000)*0.00001 #weights vector
img = np.histogram2d(unzipped_sample[0], unzipped_sample[1], bins=[15,15], range=[[-7.5,7.5], [-7.5,7.5]], normed=None, weights=w, density=None)[0]
img -= img.min()
img /= img.max()
img *= 255
cv2.imshow('do not click x', img.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
print(img[5:10,5:10])

# cv2.imshow('do not click x', 100+(img*255).astype(np.uint8))

