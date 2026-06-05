import cv2
import numpy as np


def sobel_gauss_otsu(canal, k_gauss=11, ksize=7):
    gx = cv2.Sobel(canal, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(canal, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)
    g = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    g = cv2.GaussianBlur(g, (k_gauss, k_gauss), 0)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def bordes_sv(S, V, k_gauss=11):
    sobel_s = sobel_gauss_otsu(S, k_gauss)
    sobel_v = sobel_gauss_otsu(V, k_gauss)
    return cv2.bitwise_or(sobel_v, sobel_s)


def densidad_bordes(canal, k_gauss=11):
    b = sobel_gauss_otsu(canal, k_gauss)
    return np.count_nonzero(b) / b.size