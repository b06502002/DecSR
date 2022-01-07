import numpy as np

# np.random.seed(1234321)
## This is a base profile class module
    # metroplois hasting algorithm

class BASE_PROFILE:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.beta = 3.419
        self.alpha = 1

    def uniform_proposal(self, delta=2.0):
        """
        The "MonteCarlo" part
        """
        return np.random.uniform([self.x-delta,self.y-delta], [self.x+delta, self.y+delta],size=(1,2))

    def metropolis_sampler(self, p, nsamples, proposal=uniform_proposal):
        """
        Metroplois Hastin algorithm
        """
        for i in range(nsamples):
            trial = proposal(self.x, self.y)
            acceptance = p(trial[0][0], trial[0][1])/p(self.x, self.y)
            if np.random.uniform() < acceptance:
                self.x = trial[0][0]
                self.y = trial[0][1]
            yield self.x, self.y

    def setAlpha(self, r50):
        """
        This function set the alpha value with the given relationship. According to the paper (Fast PSF modeling with DL),
        the half light radius is set as 1 pixel."""
        self.alpha = r50/np.sqrt(2**(1/(self.beta-1))-1)
        return r50/np.sqrt(2**(1/(self.beta-1))-1)

    def Moffat(self, I0):
        """
        The base of the base profile
        """
        r = np.sqrt(self.x**2+self.y**2)
        return I0*(1+(r/self.alpha)**2)**-self.beta # 2*(self.beta-1)/alpha**2*(1+(r/alpha)**2)**-self.beta

    def size_module(self, f):
        """
        R50
        """
        return f*(np.sqrt(2**(1/(self.beta-1))-1)/(2*np.sqrt(2**(1/self.beta)-1)))
