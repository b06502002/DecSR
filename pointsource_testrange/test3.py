import numpy as np
import matplotlib.pyplot as plt

# ref: https://stackoverflow.com/questions/51050658/how-to-generate-random-numbers-with-predefined-probability-distribution

def uniform_proposal(x, delta=2.0):
    return np.random.uniform(x - delta, x + delta)

def metropolis_sampler(p, nsamples, proposal=uniform_proposal):
    x = 1 # start somewhere

    for i in range(nsamples):
        trial = proposal(x) # random neighbour from the proposal distribution
        acceptance = p(trial)/p(x)

        # accept the move conditionally
        if np.random.uniform() < acceptance:
            x = trial

        yield x


def gaussian(x, mu, sigma):
    return 1./sigma/np.sqrt(2*np.pi)*np.exp(-((x-mu)**2)/2./sigma/sigma)

p = lambda x: gaussian(x, 1, 0.3) + gaussian(x, -1, 0.1) + gaussian(x, 3, 0.2)
samples = list(metropolis_sampler(p, 100000))
plt.hist(samples, bins=100)
plt.show()

# ##
# def cauchy(x, mu, gamma):
#     return 1./(np.pi*gamma*(1.+((x-mu)/gamma)**2))

# p = lambda x: cauchy(x, -2, 0.5)
# samples = list(metropolis_sampler(p, 100000))
# plt.hist(samples, bins=100)
# plt.show()

# ##
# p = lambda x: np.sqrt(x)
# samples = list(metropolis_sampler(p, 100000))
# plt.hist(samples, bins=100)
# plt.show()

# ##
# p = lambda x: (np.sin(x)/x)**2
# samples = list(metropolis_sampler(p, 100000))

# ##
# plt.hist(samples, bins=100)
# plt.show()