import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

CLASS_MAPPING = {
    "brown-glass": "glass",
    "green-glass": "glass",
    "white-glass": "glass",
    "metal":       "metal",
    "paper":       "paper",
    "cardboard":   "cardboard",
    "plastic":     "plastic",
    "biological":  "organic",
    "clothes":     "textile",
    "shoes":       "textile",
    "battery":     "battery",
    "trash":       "trash",
}

FINAL_CLASSES = sorted(set(CLASS_MAPPING.values()))

VAL_SIZE  = 0.15
TEST_SIZE = 0.15
SEED      = 42


def build_index(raw_dir: Path) -> list[dict]:
    records = []
    for src_class, tgt_class in CLASS_MAPPING.items():
        src_path = raw_dir / src_class
        if not src_path.exists():
            print(f"[WARN] carpeta no encontrada: {src_path}")
            continue
        for img_path in sorted(src_path.iterdir()):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                records.append({
                    "src_path":  img_path,
                    "src_class": src_class,
                    "tgt_class": tgt_class,
                })
    return records


def split_records(records: list[dict]) -> tuple[list, list, list]:
    labels = [r["tgt_class"] for r in records]

    train_val, test = train_test_split(
        records, test_size=TEST_SIZE,
        stratify=labels, random_state=SEED
    )
    labels_tv = [r["tgt_class"] for r in train_val]
    val_ratio  = VAL_SIZE / (1 - TEST_SIZE)

    train, val = train_test_split(
        train_val, test_size=val_ratio,
        stratify=labels_tv, random_state=SEED
    )
    return train, val, test


def copy_split(records: list[dict], split_name: str, processed_dir: Path) -> None:
    for r in records:
        dest_dir = processed_dir / split_name / r["tgt_class"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / r["src_path"].name
        if not dest_file.exists():
            shutil.copy2(r["src_path"], dest_file)


def build(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> dict:
    records = build_index(raw_dir)
    if not records:
        raise FileNotFoundError(f"No se encontraron imágenes en {raw_dir}")

    train, val, test = split_records(records)

    for split_name, split_records_ in [("train", train), ("val", val), ("test", test)]:
        copy_split(split_records_, split_name, processed_dir)

    summary = {
        "total":  len(records),
        "train":  len(train),
        "val":    len(val),
        "test":   len(test),
        "classes": FINAL_CLASSES,
    }

    print(f"Total imágenes : {summary['total']}")
    print(f"Train          : {summary['train']}")
    print(f"Val            : {summary['val']}")
    print(f"Test           : {summary['test']}")
    print(f"Clases ({len(FINAL_CLASSES)}): {FINAL_CLASSES}")

    return summary


if __name__ == "__main__":
    build()