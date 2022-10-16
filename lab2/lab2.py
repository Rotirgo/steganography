import copy
import random
import numpy as np
import skimage.metrics
from skimage.io import imsave, imshow, show, imread
from matplotlib import pyplot as plt

decompositionLvl = 3
area = 1/4

def generateW(size, seed):
    rng = np.random.default_rng(seed=seed)
    w = rng.random(size)
    return w


def insertW(f, Lvl, w, alpha):
    sizef = np.shape(f)
    partf = f[sizef[0]//(2**Lvl):sizef[0]//(2**(Lvl-1)), 0:sizef[1]//(2**Lvl)]
    sizePartf = np.shape(partf)
    fmean = np.mean(partf)
    smallfw = copy.copy(partf)
    cnt = 0
    i = 0
    d = np.abs(sizePartf[0] - sizePartf[1])
    startd = np.abs(sizePartf[0] - sizePartf[1])
    while i < len(w):
        for j in range(0, cnt+1):
            # print(i + j)
            if i + j == len(w):
                break
            if d == startd:
                smallfw[cnt-j, j] = fmean + (partf[cnt-j, j] - fmean)*(1+alpha*w[i+j])
            else:
                smallfw[sizePartf[0]-1-j, sizePartf[0]-1-cnt+j] = \
                    fmean+(partf[sizePartf[0]-1-j, sizePartf[0]-1-cnt+j] - fmean) * (1 + alpha * w[i + j])
        i += cnt+1
        if (cnt < np.min(sizePartf) - 1) & (d >= 0):
            cnt += 1
        if (cnt == np.min(sizePartf) - 1) & (d != -1):
            cnt = cnt
            d -= 1
        if d < 0:
            cnt -= 1
    fw = copy.copy(f)
    fw[sizef[0]//(2**Lvl):sizef[0]//(2**(Lvl-1)), 0:sizef[1]//(2**Lvl)] = smallfw

    # fig = plt.figure(figsize=(20, 10))
    # fig.add_subplot(1, 1, 1)
    # imshow(smallfw-partf, cmap="gray")
    # show()
    return fw


def VeyvletHaara(img):
    size = np.shape(img)

    fHorizont = np.reshape(img, ((size[0]*size[1]//2), 2))
    L = np.sum(fHorizont, axis=1)/2
    tmp = fHorizont * ([[1.0, -1.0]]*np.ones_like(fHorizont))
    H = np.sum(tmp, axis=1)/2
    L = np.reshape(L, (size[0], size[1]//2))
    H = np.reshape(H, (size[0], size[1] // 2))
    fHorizont = np.concatenate((L, H), axis=1)

    fVert = np.reshape(np.transpose(fHorizont), ((size[0] * size[1] // 2), 2))
    L = np.sum(fVert, axis=1) / 2
    tmp = fVert * ([[1.0, -1.0]] * np.ones_like(fVert))
    H = np.sum(tmp, axis=1) / 2
    L = np.reshape(L, (size[0], size[1] // 2))
    H = np.reshape(H, (size[0], size[1] // 2))
    return np.transpose(np.concatenate((L, H), axis=1))


def invVeyvletHaara(img):
    size = np.shape(img)
    resImg = copy.copy(img)

    fVert = np.zeros_like(img).astype(float)
    for i in range(0, size[0]//2):
        fVert[2*i] = img[i] + img[i+(size[0]//2)]
        fVert[2*i+1] = img[i] - img[i+(size[0]//2)]

    fHorizont = np.transpose(fVert)
    for i in range(0, size[0]//2):
        resImg[2 * i] = fHorizont[i] + fHorizont[i + (size[0] // 2)]
        resImg[2 * i + 1] = fHorizont[i] - fHorizont[i + (size[0] // 2)]
    return np.transpose(resImg)


def DVTwithLvlDecomposition(img, Lvl):
    size = np.shape(img)
    ResF = VeyvletHaara(img)
    partF = ResF[0:size[0]//2, 0:size[1]//2]
    for i in range(1, Lvl):
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


def rateW(fw, f, alpha, Lvl, size):
    sizef = np.shape(f)
    partfw = fw[sizef[0]//(2**Lvl):sizef[0]//(2**(Lvl-1)), 0:sizef[1]//(2**Lvl)]
    partf = f[sizef[0]//(2 ** Lvl):sizef[0]//(2 ** (Lvl - 1)), 0:sizef[1]//(2 ** Lvl)]

    sizePartf = np.shape(partf)
    fmean = np.mean(partf)
    w = []
    cnt = 0
    i = 0
    d = np.abs(sizePartf[0] - sizePartf[1])
    startd = np.abs(sizePartf[0] - sizePartf[1])
    while len(w) < size:
        # print(i, cnt)
        for j in range(0, cnt + 1):
            if len(w) == size:
                break
            if d == startd:
                w.append((partfw[cnt - j, j] - partf[cnt - j, j]) / (alpha*(partf[cnt - j, j]-fmean)))
                # print(f"w: {cnt - j};{j}")
            else:
                print(f"w_: {sizePartf[0]-1-j};{sizePartf[0]-1-cnt+j}")
                w.append((partfw[sizePartf[0]-1-j, sizePartf[0]-1-cnt+j] - partf[sizePartf[0]-1-j, sizePartf[0]-1-cnt+j]) /
                         (alpha*(partf[sizePartf[0]-1-j, sizePartf[0]-1-cnt+j]-fmean)))
        if (cnt < np.min(sizePartf) - 1) & (d >= 0):
            cnt += 1
        if (cnt == np.min(sizePartf) - 1) & (d != -1):
            cnt = cnt
            d -= 1
        if d < 0:
            cnt -= 1
    print(f"len: {len(w)}")
    # for i in range(0, sizePartf[0]*sizePartf[1]):  # сделать проход от низких частот к высоким
    #     w.append((partfw[i//sizePartf[1], i%sizePartf[1]] - partf[i//sizePartf[1], i%sizePartf[1]]) /
    #              (alpha*(partf[i//sizePartf[1], i%sizePartf[1]]-fmean)))
    return w


def detector(w, wnew):
    w_ = wnew[0:len(w)]
    print(w)
    print(w_)
    sum = 0
    for i in range(0, len(w)):
        sum += w[i]*w_[i]
    sum1 = np.sum(np.square(w_))
    sum2 = np.sum(np.square(w))
    # print(sum, sum2, sum1)
    delimiter = np.sum(np.square(w_)) * np.sum(np.square(w))
    p = sum/np.sqrt(delimiter)
    return p


# для визуального восприятия спектра
def contrastDecomposition(img):
    size = np.shape(img)
    contrast = copy.copy(img)
    LL = copy.copy(np.abs(contrast[0:size[0] // 2, 0:size[1]//2]))
    LH = copy.copy(np.abs(contrast[0:size[0]//2, size[1]//2:]))
    HL = copy.copy(np.abs(contrast[size[0]//2:, 0:size[1]//2]))
    HH = copy.copy(np.abs(contrast[size[0]//2:, size[1]//2:]))
    lin = np.vectorize(linary)
    contrastLL = lin(LL, np.min(LL), np.max(LL))
    contrastLH = lin(LH, np.min(LH), np.max(LH))
    contrastHL = lin(HL, np.min(HL), np.max(HL))
    contrastHH = lin(HH, np.min(HH), np.max(HH))
    contrast[0:size[0] // 2, 0:size[1]//2] = contrastLL
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
        img = 0.0
    elif img > fmax:
        img = 255.0
    else:
        img = 255*(img - fmin)/(fmax-fmin)
    return img


if __name__ == '__main__':
    C = imread("C:/Users/Никита/Desktop/стеганография/лаба2/bridge.tif").astype(int)
    #1
    sizeC = np.shape(C)
    n = int(area*(sizeC[0]//(2**decompositionLvl))*(sizeC[1]//(2**decompositionLvl)))
    W = generateW(n, 1)

    #2
    F = DVTwithLvlDecomposition(C, decompositionLvl)  # получили спектр
    # CC = invDVTwithLvlDecomposition(F, decompositionLvl)
    # print(np.average(np.abs(C-CC)))
    #3
    bestA = 0
    alpha = 0.05
    ro = 0
    roOfBest = 0
    psnrMax = 1
    psnr = 0
    # while ro <= 0.9:
    # можно находить пеовый ро > 0.9 потому что при росте а сигнал/шум падпет,
    # значит при первом таком появлении psnr будет максимален, а искажения минимальны
    for i in range(1):
        Fw = insertW(F, decompositionLvl, W, alpha)  # встроили знак в спектр
        #4
        CW = invDVTwithLvlDecomposition(Fw, decompositionLvl)  # получили изображение со встроенным знаком
        imsave("CW.png", CW)    # сохраняя в файл картинка записывается в файл не идентичная,
                                # поэтому при разности CW и savedCW в области без цвз тоже есть отличия
        #5
        savedCW = imread("CW.png")
        newFw = DVTwithLvlDecomposition(savedCW, decompositionLvl)
        #6
        newW = rateW(newFw, F, alpha, decompositionLvl, n)
        ro = detector(W, newW)
        psnr = skimage.metrics.peak_signal_noise_ratio(C, savedCW)
        if (ro > 0.9) & (psnr > psnrMax):
            bestA = alpha
            psnrMax = psnr
            roOfBest = ro
            print(f"i: {i}\tp: {ro}\tpsnr: {psnr}\ta: {alpha}")
        print(ro)
        alpha += 0.05
    print(f"p: {roOfBest}\tpsnr: {psnrMax}\tbest a: {bestA}")

    #8


    # вывод изображений
    viewF = contrastF(F, decompositionLvl)
    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(1, 2, 1)
    imshow(C, cmap="gray")
    fig.add_subplot(1, 2, 2)
    imshow(viewF, cmap="gray")

    fig2 = plt.figure(figsize=(20, 10))
    fig2.add_subplot(1, 2, 1)
    imshow(C, cmap="gray")
    fig2.add_subplot(1, 2, 2)
    imshow(savedCW.astype(float), cmap="gray")
    print(np.average(np.abs(CW - savedCW)))

    lin = np.vectorize(linary)
    difF = contrastF((Fw-newFw), decompositionLvl)
    fig3 = plt.figure(figsize=(20, 10))
    fig3.add_subplot(1, 2, 1)
    imshow(C - savedCW, cmap="gray")
    fig3.add_subplot(1, 2, 2)
    imshow(difF, cmap="gray")
    print(np.average(np.abs(W-newW[0:len(W)])))
    show()