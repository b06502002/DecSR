import numpy as np
import matplotlib.pyplot as plt
# ref: https://stackoverflow.com/questions/51050658/how-to-generate-random-numbers-with-predefined-probability-distribution
# This is 1D version

np.random.seed(1000)
def uniform_proposal(x, delta=12.0):
    return np.random.uniform(x-delta, x+delta)

def metropolis_sampler(p, nsamples, proposal=uniform_proposal):
    x = 0
    for i in range(nsamples):
        trial = proposal(x)
        acceptance = p(trial)/p(x)
        if np.random.uniform() < acceptance:
            x = trial
        yield x

def setAlpha(beta, r50):
    return r50/np.sqrt(2**(1/(beta-1))-1)

def Moffat(x, alpha, beta):
    return 2*(beta-1)/alpha**2*(1+(np.sqrt(x**2+9)/alpha)**2)**-beta

b = 3.419
p = lambda x: Moffat(x, setAlpha(b, 1), b)
samples = list(metropolis_sampler(p, 100000))[10000:]
plt.hist(samples, bins=100, range=(-10,10))
plt.show()
