import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance

PROCESSED_DIR = Path("data/processed")
TARGET        = 750
SEED          = 42

random.seed(SEED)
np.random.seed(SEED)


def flip_horizontal(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def rotate(img: Image.Image) -> Image.Image:
    angle = random.uniform(-15, 15)
    return img.rotate(angle, resample=Image.BILINEAR, expand=False)


def adjust_brightness(img: Image.Image) -> Image.Image:
    factor = random.uniform(0.8, 1.2)
    return ImageEnhance.Brightness(img).enhance(factor)


def zoom(img: Image.Image) -> Image.Image:
    w, h   = img.size
    factor = random.uniform(0.85, 1.0)
    new_w  = int(w * factor)
    new_h  = int(h * factor)
    left   = (w - new_w) // 2
    top    = (h - new_h) // 2
    img    = img.crop((left, top, left + new_w, top + new_h))
    return img.resize((w, h), Image.BILINEAR)


def add_noise(img: Image.Image) -> Image.Image:
    arr   = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 8, arr.shape)
    arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


AUGMENTATIONS = [flip_horizontal, rotate, adjust_brightness, zoom, add_noise]


def augment_one(img: Image.Image) -> Image.Image:
    ops = random.sample(AUGMENTATIONS, k=random.randint(1, 3))
    for op in ops:
        img = op(img)
    return img


def balance_train() -> None:
    split_dir = PROCESSED_DIR / "train"
    clases    = sorted([f for f in split_dir.iterdir() if f.is_dir()])

    for clase_dir in clases:
        imagenes = sorted([
            f for f in clase_dir.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}
        ])
        n = len(imagenes)

        if n > TARGET:
            elegidas = random.sample(imagenes, TARGET)
            eliminar = set(imagenes) - set(elegidas)
            for f in eliminar:
                f.unlink()
            print(f"[train] {clase_dir.name}: {n} → {TARGET} (submuestreo)")

        elif n < TARGET:
            necesarias = TARGET - n
            generadas  = 0
            idx        = 0
            while generadas < necesarias:
                src = imagenes[idx % len(imagenes)]
                img = Image.open(src).convert('RGB')
                img = augment_one(img)
                nombre = clase_dir / f"aug_{generadas:05d}_{src.name}"
                img.save(nombre)
                generadas += 1
                idx       += 1
            print(f"[train] {clase_dir.name}: {n} → {TARGET} (+{necesarias} augmented)")

        else:
            print(f"[train] {clase_dir.name}: {n} (sin cambios)")

    print("\nVal y test sin modificar (datos reales).")
    print("Balanceo completado.")


if __name__ == "__main__":
    balance_train()