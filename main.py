# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Green-2 XOR Red-3 //300 .. 298
# Cr
# 3.13
#
#
import copy

import numpy as np
from skimage.io import imsave, imshow, show, imread
from matplotlib import pyplot as plt
import math

var = 22
delta = (4 + 4*var) % 3


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def get_level(img, level):
    return (img & (2**(level-1)))


def getW(img, chanals, levels):  # возможно параметр ключа (r3 xor g2)
    arr = []
    for i in range(0,len(chanals), 1):
        if chanals[i] == 'r':
            red = (get_level(img[:, :, 0], levels[i]) >> (levels[i] - 1))
            arr.append(red)
        if chanals[i] == 'g':
            green = (get_level(img[:, :, 1], levels[i]) >> (levels[i] - 1))
            arr.append(green)
        if chanals[i] == 'b':
            blue = (get_level(img[:, :, 2], levels[i]) >> (levels[i] - 1))
            arr.append(blue)
    return arr[0] ^ arr[1]


def getCr(img):
    Red_chanal = img[:, :, 0] / 255
    Green_chanal = img[:, :, 1] / 255
    Blue_chanal = img[:, :, 2] / 255
    y = (77 * Red_chanal + 150 * Green_chanal + 29 * Blue_chanal) / 256
    Cr = (255 * (Red_chanal - y)).astype(int)
    return Cr, (255*y).astype(int)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    C = imread("C:/Users/Никита/Desktop/стенография/baboon.tif")
    W = (imread("C:/Users/Никита/Desktop/стенография/mickey.tif") / 255).astype(int)
    W2 = (imread("C:/Users/Никита/Desktop/стенография/ornament.tif") / 255).astype(int)
    CW = copy.copy(C)
    CW2 = copy.copy(C)

    # task1
    # нужные каналы
    Green_chanal = C[:, :, 1]
    Red_chanal = C[:, :, 0]

    # нужные плоскости
    greenLvl = 2
    redLvl = 3
    green2 = get_level(Green_chanal, greenLvl)
    red3 = get_level(Red_chanal, redLvl)

    # картинка на замену одной из плоскостей
    G2xorW = (green2 >> (greenLvl-1)) ^ W  # возможно надо еще один хоr с red3

    # встраивание
    mask = 255 - 2**(redLvl - 1)
    CW[:, :, 0] = ((CW[:, :, 0] & mask) | (G2xorW << (redLvl - 1)))  # для наглядности встраивания поменять сдвиг

    # task2
    # извлечение
    resW1 = getW(CW, ['g', 'r'], [greenLvl, redLvl])

    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(2, 3, 1)
    imshow(C)
    fig.add_subplot(2, 3, 2)
    imshow(W)
    fig.add_subplot(2, 3, 3)
    imshow(Red_chanal)
    fig.add_subplot(2, 3, 4)
    imshow(CW)
    fig.add_subplot(2, 3, 5)
    imshow(resW1)
    fig.add_subplot(2, 3, 6)
    imshow(G2xorW)
    show()

    # task3
    Cr, y = getCr(C)
    v = Cr % delta
    CrW = np.floor(Cr / (2 * delta)) * 2 * delta + W * delta + v

    CW2[:, :, 0] = CrW + y

    # task4
    # CrW = CW2 - y
    resW2 = (CrW - v - 2*delta*np.floor(Cr/(2*delta)))/delta

    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(2, 3, 1)
    imshow(C)
    fig.add_subplot(2, 3, 2)
    imshow(W)
    fig.add_subplot(2, 3, 3)
    imshow(v)
    fig.add_subplot(2, 3, 4)
    imshow(CW2)
    fig.add_subplot(2, 3, 5)
    imshow(resW2)
    fig.add_subplot(2, 3, 6)
    imshow(CrW)
    show()

    # external1
    ex_Cr, ex_y = getCr(CW)
    v = ex_Cr % delta
    ex_CrW = np.floor(ex_Cr / (2 * delta)) * 2 * delta + W2 * delta + v

    CW[:, :, 0] = ex_CrW + ex_y

    # resW1 = getW(CW, ['g', 'r'], [greenLvl, redLvl])
    resW3 = np.floor(ex_CrW / delta) % 2
    resW4 = getW(CW, ['g', 'r'], [greenLvl, redLvl])

    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(3, 3, 1)
    imshow(C)
    fig.add_subplot(3, 3, 2)
    imshow(W)
    fig.add_subplot(3, 3, 3)
    imshow(W2)
    fig.add_subplot(3, 3, 4)
    imshow(CW)
    fig.add_subplot(3, 3, 5)
    imshow(resW4)
    fig.add_subplot(3, 3, 6)
    imshow(resW3)
    fig.add_subplot(3, 3, 8)
    imshow(G2xorW)
    show()

    # external2
    resW4 = np.floor(CrW / delta) % 2

    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(2, 1, 1)
    imshow(W)
    fig.add_subplot(2, 1, 2)
    imshow(resW4)
    show()



    # QIM
    # при извлечении нужно CW поделить на дельту -> побочное слагаемое будет меньше 1, а второе слагаемое это W
    # затем берем по модулю 2 ->  избавились от первого слагаемого -> осталось W бинарное и побочка меньше 1
    # округляем вниз(целая часть), тогда исчезнет побочка


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
