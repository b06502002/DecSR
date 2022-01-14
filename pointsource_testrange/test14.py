import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
np.random.seed(1234321)
## This script mainly checks 
    # centroid offset

def gen_image(b=3.419, FWHM=3, E1=0.31, E2=0.11, F1=2, F2=2, G1=3, G2=2, k=0.1, offset=(-1,-2)):
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
            yield np.array((x,y))

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

    def ellip_module(e1, e2):
        ABSe = np.sqrt(e1**2+e2**2) # see also numpy.linalg.norm
        if ABSe == 0:
            return np.identity(2)
        else:
            return 1/np.sqrt(2)*np.array([[ np.sign(e2)*np.sqrt((1+ABSe)*(1+e1/ABSe)), -np.sqrt((1-ABSe)*(1-e1/ABSe))], 
                                        [ np.sqrt((1+ABSe)*(1-e1/ABSe)), np.sign(e2)*np.sqrt((1-ABSe)*(1+e1/ABSe)) ]])

    def flex_module(f1, f2, g1, g2):
        """f1 and f2 sccount for the skewness of the PSF
        while g1 and g2 accout for the triangularity"""
        F1 = -0.5*np.array([[3*f1,f2], [f2,f1]])
        F2 = -0.5*np.array([[f2,f1], [f1,3*f2]])
        G1 = -0.5*np.array([[g1,g2], [g2,-g1]])
        G2 = -0.5*np.array([[g2,-g1], [-g1,-g2]])
        return F1+G1, F2+G2

    R50 = size_module(b, FWHM)
    A = ellip_module(E1,E2)
    D1, D2 = flex_module(F1,F2,G1,G2)
    p = lambda x, y: Moffat(x, y, setAlpha(b, R50), b, 1)
    samples = list(metropolis_sampler(p, 100000))
    samples = list( np.inner(elem,R50) for elem in samples )
    samples = list( np.matmul(A,elem) + np.matmul(elem[0]*D1,elem)+np.matmul(elem[1]*D2,elem) + np.inner(k*np.linalg.norm(elem)**2,elem) for elem in samples )
    samples = list( np.add(elem,offset) for elem in samples )
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
    return img

if __name__ == "__main__":
    if os.environ['CONDA_DEFAULT_ENV'] == "tf-proj":
        gen_image()
    else:
        print("Wrong environment")
