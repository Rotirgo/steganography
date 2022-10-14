import copy
import random
import numpy as np
from skimage.io import imsave, imshow, show, imread
from matplotlib import pyplot as plt

decompositionLvl = 4
area = 1/4

def generateW(size, seed):
    np.random.seed = seed
    W = np.random.random(size)
    return W


def VeyvletHaara(img):
    size = np.shape(img)
    fHorizont = np.zeros_like(img).astype(float)
    fVert = np.zeros_like(img).astype(float)
    for i in range(0, size[0]):
        for j in range(0, size[1], 2):
            fHorizont[i][j//2] = 0.5*img[i][j] + 0.5*img[i][j+1]
            fHorizont[i][(j // 2)+(size[1]//2)] = 0.5*img[i][j] - 0.5*img[i][j + 1]

    for i in range(0, size[0], 2):
        for j in range(0, size[1]):
            fVert[i//2][j] = 0.5*fHorizont[i][j] + 0.5*fHorizont[i+1][j]
            fVert[(i//2)+(size[0]//2)][j] = 0.5*fHorizont[i][j] - 0.5*fHorizont[i+1][j]
    return fVert


def invVeyvletHaara(img):
    size = np.shape(img)
    fHorizont = np.zeros_like(img).astype(float)
    fVert = np.zeros_like(img).astype(float)
    for i in range(0, size[0]//2):
        for j in range(0, size[1]):
            fVert[2*i][j] = img[i][j] + img[i+(size[0]//2)][j]
            fVert[2*i+1][j] = img[i][j] - img[i+(size[0]//2)][j]

    for i in range(0, size[0]):
        for j in range(0, size[1]//2):
            fHorizont[i][2*j] = fVert[i][j] + fVert[i][j+(size[1]//2)]
            fHorizont[i][2*j+1] = fVert[i][j] - fVert[i][j+(size[1]//2)]
    return fHorizont


def DVTwithLvlDecomposition(img, decompositionLvl):
    size = np.shape(img)
    ResF = VeyvletHaara(img)
    partF = ResF[0:size[0]//2, 0:size[1]//2]
    for i in range(1, decompositionLvl):
        ResF[0:(size[0]//(2**i)), 0:(size[1]//(2**i))] = VeyvletHaara(partF)
        partF = ResF[0:(size[0]//(2*2**i)), 0:(size[1]//(2*2**i))]
    return ResF


def invDVTwithLvlDecomposition(img, Lvl):
    size = np.shape(img)
    ResF = copy.copy(img)
    for i in range(0, Lvl):
        partF = ResF[0:size[0] // (2 ** (Lvl - i - 1)), 0:size[1] // (2 ** (Lvl - i - 1))]
        ResF[0:size[0] // (2 ** (Lvl - i - 1)), 0:size[1] // (2 ** (Lvl - i - 1))] = invVeyvletHaara(partF)
    return ResF

# для визуального восприятия спектра
def contrastDecomposition(img):
    size = np.shape(img)
    contrast = copy.copy(img)
    LH = copy.copy(np.abs(contrast[0:size[0]//2, size[1]//2:]))
    HL = copy.copy(np.abs(contrast[size[0]//2:, 0:size[1]//2]))
    HH = copy.copy(np.abs(contrast[size[0]//2:, size[1]//2:]))
    lin = np.vectorize(linary)
    contrastLH = lin(LH, np.min(LH), np.max(LH))
    contrastHL = lin(HL, np.min(HL), np.max(HL))
    contrastHH = lin(HH, np.min(HH), np.max(HH))
    contrast[0:size[0]//2, size[1]//2:] = contrastLH
    contrast[size[0]//2:, 0:size[1]//2] = contrastHL
    contrast[size[0]//2:, size[1]//2:] = contrastHH
    return contrast


def contrastF(img, Lvl):
    size = np.shape(img)
    ResF = contrastDecomposition(img)
    LL = ResF[0:size[0] // 2, 0:size[1] // 2]
    for i in range(1, Lvl):
        ResF[0:(size[0] // (2 ** i)), 0:(size[1] // (2 ** i))] = contrastDecomposition(LL)
        LL = ResF[0:(size[0] // (2 * 2 ** i)), 0:(size[1] // (2 * 2 ** i))]
    return ResF


def linary(img, fmin, fmax):
    if img < fmin:
        img = 0
    elif img > fmax:
        img = 255
    else:
        img = 255*(img - fmin)/(fmax-fmin)
    return img


if __name__ == '__main__':
    C = imread("C:/Users/Никита/Desktop/стеганография/лаба2/bridge.tif")
    sizeC = np.shape(C)
    #test = imread("C:/Users/Никита/Desktop/стеганография/лаба2/bridge.tif")
    seed = random.seed(1)
    n = int(area*(sizeC[0]//(2**decompositionLvl))*(sizeC[1]//(2**decompositionLvl)))
    W = generateW(n, 1)

    F = DVTwithLvlDecomposition(C, decompositionLvl)  # получили спектр
    # встроили знак в спектр
    CW = invDVTwithLvlDecomposition(F, decompositionLvl)  # получили изображение со встроенным знаком

    # вывод изображений
    contrastF = contrastF(F, decompositionLvl)
    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(1, 2, 1)
    imshow(C)
    # imshow(F, cmap="gray")  # , vmin=0
    fig.add_subplot(1, 2, 2)
    imshow(contrastF, cmap="gray")





    fig1 = plt.figure(figsize=(20, 10))
    fig1.add_subplot(1, 2, 1)
    imshow(C)
    fig1.add_subplot(1, 2, 2)
    imshow(CW, cmap="gray")
    print(np.sum(np.square(C-CW)))
    show()