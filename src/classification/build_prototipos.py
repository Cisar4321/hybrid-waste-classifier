"""
Precalcula los prototipos del clasificador de MINIMA DISTANCIA (sin ML).

Se ejecuta UNA sola vez sobre el set de entrenamiento y guarda como constantes
(un .npz) los valores que el clasificador usa en tiempo de ejecucion:

  - feat_cols : orden de las 113 features.
  - mu, sigma : media y desviacion por feature (para estandarizar, z-score).
  - clases    : las 9 clases (orden alfabetico).
  - prototipos: vector promedio de las 113 features por clase (en espacio crudo).

En tiempo de ejecucion NO se vuelve a leer el train ni se entrena nada: solo se
cargan estas constantes. Reproducible con:

    python -m src.classification.build_prototipos
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = ROOT / 'results' / 'features' / 'features_train.csv'
OUT_NPZ = ROOT / 'results' / 'models' / 'prototipos_min_distancia.npz'


def construir(train_csv=TRAIN_CSV, out=OUT_NPZ):
    tr = pd.read_csv(train_csv)
    feat_cols = [c for c in tr.columns if c not in ('archivo', 'clase')]
    clases = sorted(tr['clase'].unique())

    X = tr[feat_cols].to_numpy(np.float64)
    X[~np.isfinite(X)] = 0.0
    y = tr['clase'].to_numpy()

    mu = X.mean(0)
    sigma = X.std(0)
    sigma[sigma == 0] = 1.0  # evita division por cero en features constantes

    # prototipo = vector promedio (centroide) de cada clase en espacio crudo
    prototipos = np.vstack([X[y == c].mean(0) for c in clases])

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             feat_cols=np.array(feat_cols),
             clases=np.array(clases),
             mu=mu, sigma=sigma, prototipos=prototipos)
    print(f'Guardado: {out}')
    print(f'  features={len(feat_cols)}  clases={len(clases)}  '
          f'prototipos={prototipos.shape}')
    return out


if __name__ == '__main__':
    construir()
