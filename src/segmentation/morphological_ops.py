import cv2
import numpy as np


def kernels_adaptativos(mask, frac_open=0.015, frac_close=0.05):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return 3, 7
    area_obj = stats[1:, cv2.CC_STAT_AREA].max()
    lado = np.sqrt(area_obj)
    k_open = max(3, int(lado * frac_open) | 1)
    k_close = max(5, int(lado * frac_close) | 1)
    return k_open, k_close


def relleno_exitoso(cerrado, relleno, factor=1.5):
    area_bordes = np.count_nonzero(cerrado)
    area_relleno = np.count_nonzero(relleno)
    if area_bordes == 0:
        return False
    return area_relleno > factor * area_bordes


def rellenar_contornos(mask, min_area=500, pad=20, max_intentos=4, paso=6):
    k_open, k_close = kernels_adaptativos(mask)

    for _ in range(max_intentos):
        m = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

        ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        abierto = cv2.morphologyEx(m, cv2.MORPH_OPEN, ko)

        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        cerrado = cv2.morphologyEx(abierto, cv2.MORPH_CLOSE, kc)

        contornos, _ = cv2.findContours(cerrado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        relleno = np.zeros_like(m)
        for c in contornos:
            if cv2.contourArea(c) >= min_area:
                cv2.drawContours(relleno, [c], -1, 255, thickness=cv2.FILLED)

        if relleno_exitoso(cerrado, relleno):
            break
        k_close = (k_close + paso) | 1

    abierto = abierto[pad:-pad, pad:-pad]
    cerrado = cerrado[pad:-pad, pad:-pad]
    relleno = relleno[pad:-pad, pad:-pad]
    return abierto, cerrado, relleno, k_open, k_close