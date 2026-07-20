"""
Clasificador clasico por MINIMA DISTANCIA EUCLIDIANA a un prototipo por clase
(sin machine learning, sin sklearn, sin entrenamiento en tiempo de ejecucion).

Metodo (reconocimiento de patrones clasico, "nearest centroid"):
  1. Cada clase se representa por su PROTOTIPO = el vector promedio de las 113
     features del set de entrenamiento (un centroide por clase).
  2. Para clasificar una imagen nueva se calcula la distancia euclidiana de su
     vector de features al prototipo de cada clase y gana la clase mas cercana
     (argmin de la distancia).

Estandarizacion (z-score): las 113 features tienen escalas muy distintas
(p. ej. contraste GLCM ~cientos frente a circularidad ~0-1). Sin normalizar,
la distancia euclidiana quedaria dominada por las features de mayor escala.
Por eso, antes de medir distancias, cada feature se estandariza con la media y
la desviacion del train (constantes precalculadas). Es la misma distancia
euclidiana, pero en el espacio normalizado — la forma correcta del clasificador
de minima distancia. Equivale a una distancia de Mahalanobis diagonal.

Los prototipos, la media y la desviacion se calculan UNA sola vez con
`build_prototipos.py` y se guardan como constantes en
`results/models/prototipos_min_distancia.npz`. Aqui solo se cargan.

Las features usan la escala de OpenCV (H in [0,179]; S,V,L,a,b in [0,255]).
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROTO_NPZ = ROOT / 'results' / 'models' / 'prototipos_min_distancia.npz'

_CACHE = None


def _cargar():
    """Carga (y cachea) los prototipos precalculados desde el .npz."""
    global _CACHE
    if _CACHE is None:
        d = np.load(PROTO_NPZ, allow_pickle=True)
        _CACHE = {
            'feat_cols': [str(c) for c in d['feat_cols']],
            'clases': [str(c) for c in d['clases']],
            'mu': d['mu'].astype(np.float64),
            'sigma': d['sigma'].astype(np.float64),
            'prototipos': d['prototipos'].astype(np.float64),
        }
    return _CACHE


# Orden alfabetico de clases = mismo orden que usa el resto del pipeline.
CLASES = _cargar()['clases']


def _vector(feat, feat_cols):
    """Arma el vector de 113 features en el orden esperado desde un dict/Series."""
    vec = np.empty(len(feat_cols), dtype=np.float64)
    for i, c in enumerate(feat_cols):
        try:
            v = float(feat.get(c, 0.0))
        except (TypeError, ValueError):
            v = 0.0
        vec[i] = 0.0 if v != v else v  # NaN -> 0
    return vec


def distancias(feat):
    """Distancia euclidiana (estandarizada) del vector de features a cada prototipo.

    Retorna dict {clase: distancia}.
    """
    m = _cargar()
    x = (_vector(feat, m['feat_cols']) - m['mu']) / m['sigma']
    proto_z = (m['prototipos'] - m['mu']) / m['sigma']          # (9, 113)
    d = np.sqrt(((proto_z - x[None, :]) ** 2).sum(axis=1))       # (9,)
    return {c: float(dist) for c, dist in zip(m['clases'], d)}


def probabilidades(dists):
    """Pseudo-distribucion para graficar: mayor cercania -> mayor peso.

    No son probabilidades calibradas; se usa la inversa de la distancia
    normalizada (softmin) solo para visualizar la 'confianza' relativa.
    """
    eps = 1e-9
    sims = {c: 1.0 / (dst + eps) for c, dst in dists.items()}
    total = sum(sims.values())
    return {c: s / total for c, s in sims.items()}


def clasificar_min_distancia(feat):
    """Clasifica un vector de features por minima distancia al prototipo.

    Parametros
    ----------
    feat : dict | pandas.Series
        Mapea nombre_de_feature -> valor (mismas 113 claves que el CSV).

    Retorna
    -------
    clase : str                      clase del prototipo mas cercano
    proba : dict {clase: pseudo-probabilidad}
    dists : dict {clase: distancia euclidiana estandarizada}
    """
    dists = distancias(feat)
    clase = min(dists, key=dists.get)
    return clase, probabilidades(dists), dists
