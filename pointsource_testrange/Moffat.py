import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def base(I_0,alpha,beta,r):
    return I_0*(1+r/alpha)**(-beta)

def main(I_0,alpha,beta,r):
    print(I_0*(1+r/alpha)**(-beta))

if __name__ == "__main__":
    # main(100,1,1,2)
    pass