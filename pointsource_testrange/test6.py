import numpy as np
import matplotlib.pyplot as plt
import cv2

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

def Moffat(x, y, alpha, beta):
    r = np.sqrt(x**2+y**2)
    return 2*(beta-1)/alpha**2*(1+(r/alpha)**2)**-beta

b = 3.419
p = lambda x, y: Moffat(x, y, setAlpha(b, 1), b)
samples = list(metropolis_sampler(p, 100000))
unzipped_object = zip(*samples)
unzipped_sample = list(unzipped_object)

img = np.histogram2d(unzipped_sample[0], unzipped_sample[1], bins=[32, 32], range=[[-16,16], [-16,16]], normed=None, weights=None, density=None)
cv2.imshow('do not click x', img)
cv2.waitKey(0)
cv2.destroyAllWindow()
