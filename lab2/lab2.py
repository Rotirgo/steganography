import copy
import random
import numpy as np
from skimage.io import imsave, imshow, show, imread
from matplotlib import pyplot as plt


def discretVeyvletTransformation(img, decompositionLvl):
    size = np.shape(img)
    f = copy.copy(img)


    return f


if __name__ == '__main__':
    C = imread("C:/Users/Никита/Desktop/стеганография/лаба2/bridge.tif")

    F = discretVeyvletTransformation(C, 3)

    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(1, 1, 1)
    imshow(C)
    show()