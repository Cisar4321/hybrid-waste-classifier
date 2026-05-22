import cv2
import numpy as np
from pathlib import Path


TARGET_SIZE = (256, 256)
CLIP_LIMIT  = 2.0
TILE_SIZE   = (8, 8)


def resize(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    # img en BGR (formato OpenCV)
    lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_SIZE)
    l     = clahe.apply(l)
    lab   = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess(img: np.ndarray) -> np.ndarray:
    img = resize(img)
    img = apply_clahe(img)
    return img


def preprocess_from_path(img_path: Path) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"No se pudo leer: {img_path}")
    return preprocess(img)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_path = Path("data/processed/train/cardboard")
    img_path  = next(test_path.iterdir())

    original  = cv2.imread(str(img_path))
    procesada = preprocess(original)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(procesada, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Resize + CLAHE')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()