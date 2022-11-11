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


def viewImgs(imgs, title, steps, *args):
    for i in range(0, len(imgs)):
        fig = plt.figure(figsize=(7, 7))
        plt.imshow(imgs[i], cmap="gray")
        if len(args) > 0:
            plt.title(f"{title}{steps[i]}{args[0]}")
        else:
            plt.title(f"{title}{steps[i]}")
    show()


def printTable(table, coltype, rowtype):
    first_row = "\t\t\t|\t"
    delimiter = "-"*9
    for i in range(0, len(table)):
        first_row += f"s: {coltype[i]}\t|\t"
    delimiter += "-"*(len(first_row) - 1 + 4*len(table))
    print(first_row)
    print(delimiter)
    i = 0
    for row in table:
        text = f"JPEG: {rowtype[i]}\t|\t"
        for el in row:
            text += f"{el:.3f}\t|\t"
        i += 1
        print(text)
        print(delimiter)


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
    alpha = 0.9  # 0.81 нет смысла брать большое(>25), так как яркость ограничивается [0, 255]
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
    print(f"Without changes:\t\tp: {ro:.5f}\t\tpsnr: {psnr:.3f}\t\ta: {alpha}\n\n")

    # циклический сдвиг с r от 0.1 до 0.9 с шагом 0.1 (сдвиг выполняется на r*N пикселей)
    cyclyImgs = []
    cycleShift = np.arange(0.1, 1, 0.1)
    cycle_ro = []
    for shift in cycleShift:
        deformImg = cycle_shift(savedCW, shift)
        cyclyImgs.append(deformImg)
        newFw = lab2.DVTwithLvlDecomposition(deformImg, decompositionLvl)
        newW = lab2.rateW(newFw, F, alpha, decompositionLvl, n, 0)
        ro = lab2.detector(W, newW)
        cycle_ro.append(ro)
        psnr = skimage.metrics.peak_signal_noise_ratio(savedCW, deformImg)
        print(f"Cycle shift {shift:.1f}:\t\tp: {ro:.5f}\t\tpsnr: {psnr:.3f}\t\ta: {alpha}")
    print("\n\n")

    # поворот от 1 до 90 с шагом 8.9
    rotationImgs = []
    rotation = np.arange(1, 91, 8.9)
    rotation_ro = []
    for rot in rotation:
        deformImg = rotate(savedCW, rot)
        rotationImgs.append(deformImg)
        newFw = lab2.DVTwithLvlDecomposition(deformImg, decompositionLvl)
        newW = lab2.rateW(newFw, F, alpha, decompositionLvl, n, 0)
        ro = lab2.detector(W, newW)
        rotation_ro.append(ro)
        psnr = skimage.metrics.peak_signal_noise_ratio(savedCW, deformImg)
        print(f"Rotation {rot:.1f}:\t\tp: {ro:.5f}\t\tpsnr: {psnr:.3f}\t\ta: {alpha}")
    print("\n\n")

    # гауссовское размытие сигма от 1 до 4 с шагом 0.5 (написать свертку CW равно g**C)
    GaussBlurImgs = []
    GaussBlur = np.arange(1, 4.1, 0.5)
    GaussBlur_ro = []
    for sigma in GaussBlur:
        deformImg = Gauss_blur(savedCW, sigma)
        GaussBlurImgs.append(deformImg)
        newFw = lab2.DVTwithLvlDecomposition(deformImg, decompositionLvl)
        newW = lab2.rateW(newFw, F, alpha, decompositionLvl, n, 0)
        ro = lab2.detector(W, newW)
        GaussBlur_ro.append(ro)
        psnr = skimage.metrics.peak_signal_noise_ratio(savedCW, deformImg)
        print(f"Gaussian Blur {sigma}:\t\tp: {ro:.5f}\t\tpsnr: {psnr:.3f}\t\ta: {alpha}")
    print("\n\n")

    # jpeg качество от 30 до 90 с шагом 10
    jpegImgs = []
    jpeg = np.arange(30, 100, 10)
    jpeg_ro = []
    for QF in jpeg:
        deformImg = jpegFunc(savedCW, QF)
        jpegImgs.append(deformImg)
        newFw = lab2.DVTwithLvlDecomposition(deformImg, decompositionLvl)
        newW = lab2.rateW(newFw, F, alpha, decompositionLvl, n, 0)
        ro = lab2.detector(W, newW)
        jpeg_ro.append(ro)
        psnr = skimage.metrics.peak_signal_noise_ratio(savedCW, deformImg)
        print(f"JPEG {QF}:\t\tp: {ro:.5f}\t\tpsnr: {psnr:.3f}\t\ta: {alpha}")
    print("\n\n")

    # во второй лабы мы встраивали с помощью вейвлета Хаара.
    # он основан на выделении границ
    # при изменении положения границ объектов спектр тоже уедет(согласно преобразованию изображения)

    # при циклическом сдвиге форма границ объектов остается, но они смещаются => спектр тоже сместиться
    # и цвз надо искать в другом месте, а не там где мы вставляли
    # результат спектр ДВП изменен, цвз потеряна

    # при повороте границы тоже повернуться, а значит изменяться вертикальные и горизонтальные объекты, что приведет
    # к изменеию спектра в ДВП
    # новые горизонтальные и вертикальные составляющие + цвз кусочно остается в месте встраивания
    # результат спектр искажен, цвз утеряна

    # размытие просто уменьшает резкость границ -> спектр сглаживается, но границы остаются и цвз остается на месте
    # но размытие сглаживает -> цвз имеет маленькие изменения в спектре, значит оно затирается
    # но если цвз имеет значительные значения(например видимое цвз), то сглаживание что-то оставит, а цвз обнаружиться

    # в jpeg границы остаются и цвз остается на месте
    # при расжатии картинки с появляются потери из-за "типо квантования" -> в областях 8х8 происходит обрезание яркости
    # в областях 8х8 где мало различные пиксели происходит сглаживание(псевдо усреднение)
    # в областях 8х8 где значительно различные пиксели впринципе границы объектов остаются
    # вывод: при малых значениях цвз детектирование может быть затруднено
    # из-за обрезок->локальное сглаживание->сглаживание спектра
    # при значительных значения цвз детектирование должно быть получше

    print(f"means:\ncycle shift: {np.mean(cycle_ro)}\n"
          f"rotation: {np.mean(rotation_ro)}\n"
          f"Gaussian Blur: {np.mean(GaussBlur_ro)}\n"
          f"JPEG: {np.mean(jpeg_ro)}")

    # viewImgs(cyclyImgs, "cycle shift with r=", cycleShift)
    # viewImgs(rotationImgs, "rotation with ", rotation, "°")
    # viewImgs(GaussBlurImgs, "Gaussian Blur with sigma=", GaussBlur)
    # viewImgs(jpegImgs, "JPEG with QF=", jpeg)

    # два зашумления на выбор составит таблицу устойчивости по двум параметрам
    # (1 параметр шума - строки, 2 параметр шума - столбцы)
    # jpeg и GaussBlur
    twiceDeformImgs = []
    firstDeform = jpeg
    secondDeform = GaussBlur
    roTable = np.zeros((len(firstDeform), len(secondDeform)))
    for blur in secondDeform:
        for QF in firstDeform:
            deformImg = Gauss_blur(savedCW, blur)  # просто поменять функции чтоб другие посмотреть.
            deformImg = jpegFunc(deformImg, QF)
            twiceDeformImgs.append(deformImg)
            newFw = lab2.DVTwithLvlDecomposition(deformImg, decompositionLvl)
            newW = lab2.rateW(newFw, F, alpha, decompositionLvl, n, 0)
            ro = lab2.detector(W, newW)
            roTable[list(firstDeform).index(QF), list(secondDeform).index(blur)] = ro
            psnr = skimage.metrics.peak_signal_noise_ratio(savedCW, deformImg)

    printTable(roTable, jpeg, GaussBlur)
