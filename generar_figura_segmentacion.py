"""Genera una figura de ejemplos de segmentacion, una fila por clase.
Usa el mismo pipeline del notebook 02 (bilateral condicional -> Sobel S|V -> Otsu -> morfologia)."""
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Dev\UTEC\Procesamiento_Imagenes\Proyecto\hybrid-waste-classifier")
sys.path.append(str(ROOT))
from src.segmentation.edge_detection import bordes_sv, densidad_bordes
from src.segmentation.morphological_ops import rellenar_contornos

TRAIN = ROOT / "data" / "processed" / "train"
OUT = ROOT / "results" / "figures" / "02_ejemplos_segmentacion_por_clase.png"
UMBRAL_DENSIDAD = 0.15
K_GAUSS = 11
EXT = {".jpg", ".jpeg", ".png"}

clases = sorted(d.name for d in TRAIN.iterdir() if d.is_dir())
fig, axes = plt.subplots(len(clases), 4, figsize=(14, 3.2 * len(clases)))
cols = ["Original", "Bordes Sobel S|V", "Mascara (relleno)", "Overlay"]

for i, clase in enumerate(clases):
    img_path = next(p for p in sorted((TRAIN / clase).iterdir()) if p.suffix.lower() in EXT)
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    hsv_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S_raw = cv2.GaussianBlur(hsv_raw[:, :, 1], (5, 5), 0)
    densidad = densidad_bordes(S_raw, K_GAUSS)
    img_lp = cv2.bilateralFilter(img_bgr, 50, 220, 300) if densidad > UMBRAL_DENSIDAD else img_bgr.copy()

    H, S, V = cv2.split(cv2.cvtColor(img_lp, cv2.COLOR_BGR2HSV))
    sobel_comb = bordes_sv(S, V, K_GAUSS)
    _, _, relleno, _, _ = rellenar_contornos(sobel_comb)

    overlay = img_rgb.copy()
    overlay[relleno > 0] = (0.5 * overlay[relleno > 0] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    cobertura = 100 * np.count_nonzero(relleno) / relleno.size

    panels = [img_rgb, sobel_comb, relleno, overlay]
    cmaps = [None, "gray", "gray", None]
    for j, (p, cm) in enumerate(zip(panels, cmaps)):
        axes[i, j].imshow(p, cmap=cm)
        axes[i, j].axis("off")
        if i == 0:
            axes[i, j].set_title(cols[j], fontsize=12, fontweight="bold")
    axes[i, 0].set_ylabel(clase, fontsize=12, fontweight="bold", rotation=0, labelpad=45, va="center")
    axes[i, 0].axis("on"); axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
    axes[i, 3].text(5, 20, f"cobertura {cobertura:.0f}%", color="yellow", fontsize=10, fontweight="bold")

plt.suptitle("Ejemplos de segmentacion por clase (pipeline HSV + Sobel + morfologia)", fontsize=14)
plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=130, bbox_inches="tight")
print("Guardado:", OUT)
