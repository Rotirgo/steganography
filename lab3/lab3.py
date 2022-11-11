from scipy.signal import convolve2d

import lab2.lab2 as lab2
from skimage.io import imsave, imshow, show, imread
import numpy as np
import skimage.metrics
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import copy

decompositionLvl = 3


def cycle_shift(img, r):
    size = np.shape(img)
    new_img = np.roll(img, int(r * size[1]), axis=1)
    new_img = np.roll(new_img, int(r * size[0]), axis=0)
    return new_img


def rotate(img, angle):
    new_img = Image.fromarray(img, mode="L")
    new_img = np.asarray(new_img.rotate(angle))
    return new_img


def Gauss_blur(img, s):
    M = 2 * int(3 * s) + 1
    impulse_characteristic = np.vectorize(calc_g, signature="(),(n),()->(k)")
    g = impulse_characteristic(m1=np.arange(0, M, 1), m2=np.arange(0, M, 1), sigma=s)
    g /= np.sum(g)
    new_img = convolve2d(img, g, mode="same")
    return new_img


def calc_g(m1, m2, sigma):
    M = 2 * np.floor(3 * sigma) + 1
    tmp1 = (m1 - M / 2) ** 2
    tmp2 = (m2 - M / 2) ** 2
    power = -(tmp1 + tmp2) / (2 * sigma)
    GaussH = np.exp(power)
    return GaussH


def jpegFunc(img, quality):
    imsave("tmpFile.jpeg", img, quality=quality)
    new_img = imread("tmpFile.jpeg")
    return new_img


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    # C = imread("C:/Users/Никита/Desktop/стеганография/лаба2/bridge.tif").astype(int)  # alpha = 0.57
    C = imread("C.png")  # alpha = 0.8

    sizeC = np.shape(C)
    n = int(0.25 * (sizeC[0] // (2 ** decompositionLvl)) * (sizeC[1] // (2 ** decompositionLvl)))
    W = lab2.generateW(n, 1)

    # встраивание
    F = lab2.DVTwithLvlDecomposition(C, decompositionLvl)
    alpha = 0.81
    Fw = lab2.insertW(F, decompositionLvl, W, alpha, 0)  # встроили знак в спектр
    CW = lab2.invDVTwithLvlDecomposition(Fw, decompositionLvl)  # получили изображение со встроенным знаком
    imsave("CW.png", CW)

    # работа со встроенным цвз
    # без искажений
    savedCW = imread("CW.png")
    newFw = lab2.DVTwithLvlDecomposition(savedCW, decompositionLvl)
    newW = lab2.rateW(newFw, F, alpha, decompositionLvl, n, 0)
    ro = lab2.detector(W, newW)
    psnr = skimage.metrics.peak_signal_noise_ratio(C, savedCW)
    print(f"Without changes:\t\tp: {ro:.5f}\tpsnr: {psnr:.3f}\tbest a: {alpha}")

    # циклический сдвиг с r от 0.1 до 0.9 с шагом 0.1 (сдвиг выполняется на r*N пикселей)
    cyclyImgs = []
    cycleShift = np.arange(0.1, 1, 0.1)
    for shift in cycleShift:
        cyclyImgs.append(cycle_shift(savedCW, shift))

    # поворот от 1 до 90 с шагом 8.9
    rotationImgs = []
    rotation = np.arange(1, 91, 8.9)
    for rot in rotation:
        rotationImgs.append(rotate(savedCW, rot))

    # гауссовское размытие сигма от 1 до 4 с шагом 0.5 (написать свертку CW равно g**C)
    GaussBlurImgs = []
    GaussBlur = np.arange(1, 4.1, 0.5)
    for sigma in GaussBlur:
        GaussBlurImgs.append(Gauss_blur(savedCW, sigma))

    # jpeg качество от 30 до 90 с шагом 10
    jpegImgs = []
    jpeg = np.arange(30, 100, 10)
    for QF in jpeg:
        jpegImgs.append(jpegFunc(savedCW, QF))

    # два зашумления на выбор составит таблицу устойчивости по двум параметрам
    # (1 параметр шума - строки, 2 параметр шума - столбцы)
