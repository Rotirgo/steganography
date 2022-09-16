# Green-2 XOR Red-3 //300 .. 298
# Cr
# 3.13

import copy
import random
import numpy as np
from skimage.io import imsave, imshow, show, imread
from matplotlib import pyplot as plt

var = 22
delta = 4 + 4*(var % 3)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def get_level(img, level):
    return (img & (2**(level-1)))


def getW(img, chanals, levels):
    arr = []
    for i in range(0, len(chanals), 1):
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
    y = (255*y).astype(int)
    Cr = img[:, :, 0] - y
    return Cr, y


def border_processing_function(element_value):
    if element_value > 255:
        return 255
    elif element_value < 0:
        return 0
    else:
        return element_value


def border_processing(img):
    vector_img = np.vectorize(border_processing_function)
    new_img = vector_img(img)
    return new_img


def textIntoImg(text, img, p, seed):
    binText = ''.join(format(x, '08b') for x in bytearray(text, 'utf-8'))
    binl = f'{len(binText):b}'
    while len(binl) < 32:
        binl = '0' + binl
    binData = binl + binText

    a = np.arange(0, len(img)**2, 1)
    random.shuffle(a, random.seed(seed))
    textImg = np.zeros(np.shape(img)).astype(int)
    loc = calcPosition(a)
    for i in range(0, len(binData)):
        textImg[loc[0][i], loc[1][i]] = int(binData[i])
    img = (img & (255-2**(p-1))) | (textImg << (p-1))
    return img, textImg


def bits2char(arr):
    intVal = 0
    for i in range(-len(arr), 0, 1):
        intVal += (2**(abs(i)-1))*int(arr[i])
    return chr(intVal)


def getTextFromImg(img, p, seed):
    imgW = ((img & (2 ** (p - 1))) >> (p - 1)).astype(int)
    w = ''
    a = np.arange(0, len(img) ** 2, 1)
    random.shuffle(a, random.seed(seed))
    loc = calcPosition(a)
    size = 0
    for i in range(0, 32):
        if imgW[loc[0][i], loc[1][i]] == 1:
            size += 2 ** (31 - i)
    for i in range(32, size+32):
        w += str(imgW[loc[0][i], loc[1][i]])
    w = np.reshape(list(w), (int(len(w) / 8), 8)).astype(int)
    text = ''
    for row in w:
        text += bits2char(row)
    return text


def calcPositionFunction(n):
    j = n % 512
    i = (n - j)/512
    return int(i), int(j)


def calcPosition(arr):
    vector = np.vectorize(calcPositionFunction)
    vectorPositions = vector(arr)
    return vectorPositions


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    C = imread("C:/Users/Никита/Desktop/стенография/baboon.tif")
    W = (imread("C:/Users/Никита/Desktop/стенография/mickey.tif") / 255).astype(int)
    W2 = (imread("C:/Users/Никита/Desktop/стенография/ornament.tif") / 255).astype(int)
    CW = copy.copy(C)
    CW2 = copy.copy(C)
    C2 = imread("C:/Users/Никита/Desktop/стенография/goldhill.tif")

    print(delta)
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
    G2xorR3 = (green2 >> (greenLvl-1)) ^ (red3 >> (redLvl-1))  # C
    G2xorR3xorW = (green2 >> (greenLvl-1)) ^ (red3 >> (redLvl-1)) ^ W

    # встраивание
    mask = 255 - 2**(redLvl - 1)
    # замена плоскости
    # CW[:, :, 0] = ((CW[:, :, 0] & mask) | (G2xorR3xorW << (redLvl - 1)))  # для наглядности встраивания поменять сдвиг
    # xor с плоскостью
    CW[:, :, 0] = (CW[:, :, 0] & mask) | ((red3 >> (redLvl-1)) ^ W) << (redLvl-1)

    # task2
    # извлечение
    resW1 = getW(CW, ['g', 'r'], [greenLvl, redLvl])
    resW1 = resW1 ^ G2xorR3

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
    imshow(G2xorR3xorW)

    # task3
    Cr, y = getCr(C)
    v = Cr % delta
    CrW = np.floor(Cr / (2 * delta)) * 2 * delta + W * delta + v  # 140

    tmp = border_processing(CrW + y)
    CW2[:, :, 0] = tmp

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

    # external1
    # G2xorR3v2 = ((CW[:, :, 1] & (2 ** (greenLvl - 1))) >> (greenLvl - 1)) ^ (
    #             (CW[:, :, 0] & (2 ** (redLvl - 1))) >> (redLvl - 1))
    copyCW = copy.copy(CW)
    ex_Cr, ex_y = getCr(CW)
    v = ex_Cr % delta
    ex_CrW = np.floor(ex_Cr / (2 * delta)) * 2 * delta + W2 * delta + v

    tmp = border_processing(ex_CrW + ex_y)
    CW[:, :, 0] = tmp

    resW3 = np.floor(ex_CrW / delta) % 2
    resW4 = getW(CW, ['g', 'r'], [greenLvl, redLvl])
    resW4 = resW4 ^ G2xorR3

    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(3, 3, 1)
    imshow(copyCW)
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
    imshow(G2xorR3)

    # СВИ-4: извлечение не зная С
    resW4 = np.floor(CrW / delta) % 2

    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(2, 1, 1)
    imshow(W)
    fig.add_subplot(2, 1, 2)
    imshow(resW4)

    # external2
    text = "Look, I was gonna go easy on you not to hurt your feelings.\"\n" \
           "\"But I'm only going to get this one chance.\"\n" \
           "\"Something's wrong, I can feel it.\"\n" \
           "Six minutes. Six Minutes. Six minutes, Slim Shady, you're on!\n" \
           "\"Just a feeling I've got. Like something's about to happen,\n" \
           "but I don't know what.\n" \
           "If that means what I think it means, we're in trouble,\n" \
           "big trouble; and if he is as bananas as you say,\n" \
           "I'm not taking any chances.\"\n" \
           "\"You are just what the doc ordered.\""
    seed = 1
    p = 1
    CWbin, textImg = textIntoImg(text, copy.copy(C2), p, seed)

    newText = getTextFromImg(CWbin, p, seed)
    print(newText)

    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(2, 2, 1)
    imshow(C2)
    fig.add_subplot(2, 2, 2)
    imshow(textImg)
    fig.add_subplot(2, 2, 3)
    imshow(CWbin, cmap='gray', vmin=C2.min(), vmax=C2.max())
    show()


    # QIM
    # при извлечении нужно CW поделить на дельту -> побочное слагаемое будет меньше 1, а второе слагаемое это W
    # затем берем по модулю 2 ->  избавились от первого слагаемого -> осталось W бинарное и побочка меньше 1
    # округляем вниз(целая часть), тогда исчезнет побочка


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
