import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import exe.base_profile as Bp
# This script checks the utility function (base profile)
np.random.seed(1234321)

BP = Bp.BASE_PROFILE()
BP.beta = 3.419 # beta in the base profile
FWHM = 4
R50 = BP.size_module(FWHM)
BP.setAlpha(R50)
p = lambda : BP.Moffat(1)
samples = list(BP.metropolis_sampler(p, 100000))
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
