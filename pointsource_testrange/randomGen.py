import os
import numpy as np
import cv2
np.random.seed(20220114)

# this script generates parameters for distorting the point source. the parameters are sampled from a hypercube
# Be extra careful here! Notice that the skewness, triangularity and kurtosis here are in the same unit as FWHM

def main(offL=-0.5, offU=0.5, fwhmL=1.5, fwhmU=6, eL=-0.25, eU=0.25, skL=-0.25, skU=0.25, trL=-0.1, trU=0.1, kL=-0.4, kU=0.4):
    NumOfImg = 100000
    NPY = np.array([np.array([np.random.rand()-0.5, np.random.rand()-0.5, (np.random.rand()*4.5)+1.5, (np.random.rand()-0.5)/2, (np.random.rand()-0.5)/2, (np.random.rand()-0.5)/2, (np.random.rand()-0.5)/2, (np.random.rand()-0.5)/5, (np.random.rand()-0.5)/5, (np.random.rand()-0.5)*4/5]) for ii in range(NumOfImg)])

    # inside the inner np array it looks like [offset1, offset2, FWHM, ellipx, ellipy, skewx, skewy, triax, triay, kurto]
    # the total numpy array looks like:
    # array([[offset1, offset2, FWHM, ellipx, ellipy, skewx, skewy, triax, triay, kurto],
    #        [offset1, offset2, FWHM, ellipx, ellipy, skewx, skewy, triax, triay, kurto],
    #  ....   
    #        [offset1, offset2, FWHM, ellipx, ellipy, skewx, skewy, triax, triay, kurto]])

    np.save("./pointsource_testrange/genNPY/rawparam.npy", NPY)
    NPY[:,9] = np.multiply(NPY[:,9], np.square(NPY[:,2]))
    NPY[:,7] = np.multiply(NPY[:,7], NPY[:,2])
    NPY[:,8] = np.multiply(NPY[:,8], NPY[:,2])
    NPY[:,5] = np.multiply(NPY[:,5], NPY[:,2])
    NPY[:,6] = np.multiply(NPY[:,6], NPY[:,2])

    # np.save("./pointsource_testrange/genNPY/dimLess_param.npy", NPY)
    return None


if __name__ == "__main__":
    modeForGen = input("Custom range? (y/n): ")
    if os.environ['CONDA_DEFAULT_ENV'] == "tf-proj":
        if modeForGen=="y":
            # fwhml = input("Please insert a lower-bound value for FWHM: ")
            # fwhmu = input("Please insert an upper-bound value for FWHM: ")
            # el = input("Please insert a lower-bound value for ellipticity: ")
            # eu = input("Please insert a upper-bound value for ellipticity: ")
            # skl = input("Please insert a lower-bound value for skewness: ")
            # sku = input("Please insert a upper-bound value for skewness: ")
            # trl = input("Please insert a lower-bound value for triangularity: ")
            # tru = input("Please insert a upper-bound value for triangularity: ")
            # kl = input("Please insert a lower-bound value for kurtosis: ")
            # ku = input("Please insert a upper-bound value for kurtosis: ")
            # offl = input("Please insert a lower-bound value for offset: ")
            # offu = input("Please insert a upper-bound value for offset: ")
            # main(offl, offu, fwhml, fwhmu, el, eu, skl, sku, trl, tru, kl, ku)
            print("this version has not done yet!")   
            # NPY = np.load("./pointsource_testrange/genNPY/rawparam.npy")
            # NPY[:,9] = np.divide(NPY[:,9], np.square(NPY[:,2]))
            # NPY[:,7] = np.divide(NPY[:,7], NPY[:,2])
            # NPY[:,8] = np.divide(NPY[:,8], NPY[:,2])
            # NPY[:,5] = np.divide(NPY[:,5], NPY[:,2])
            # NPY[:,6] = np.divide(NPY[:,6], NPY[:,2])
            # np.save("./pointsource_testrange/genNPY/GenImg_param.npy",NPY)
        elif modeForGen=="n":
            main()
    else:
        print("Wrong environment")