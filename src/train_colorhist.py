from __future__ import annotations
import os
import glob
from typing import List, Tuple

import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

from features import extract_features

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "colorhist_knn.joblib")

IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp")


def _find_images(label_dir: str) -> List[str]:
    paths: List[str] = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(label_dir, ext)))
    return paths


def _load_dataset(root: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X: List[np.ndarray] = []
    y: List[str] = []
    classes: List[str] = []

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Expected data directory not found: {root}")

    for label in sorted(os.listdir(root)):
        label_dir = os.path.join(root, label)
        if not os.path.isdir(label_dir):
            continue
        classes.append(label)
        files = _find_images(label_dir)
        for fp in files:
            try:
                with Image.open(fp) as img:
                    feats = extract_features(img)
                    X.append(feats)
                    y.append(label)
            except Exception:
                pass

    if not X:
        raise RuntimeError(f"No images found under {root}")

    return np.stack(X), np.array(y), classes


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading training data...")
    X, y, _classes = _load_dataset(TRAIN_DIR)
    print(f"Samples: {len(y)} across {len(set(y))} classes")

    print("Splitting train/holdout...")
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    print("Training k-NN model...")
    clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
    clf.fit(x_train, y_train)

    print("Evaluating...")
    ypred = clf.predict(x_holdout)
    acc = accuracy_score(y_holdout, ypred)
    print(f"Holdout accuracy: {acc:.3f}")
    
    try:
        print(classification_report(y_holdout, ypred))
    except Exception:
        pass

    print(f"Saving model to {MODEL_PATH}")
    joblib.dump({
        "model": clf,
        "classes": clf.classes_,
        "bins_per_channel": 16,
    }, MODEL_PATH)

    print("Training complete")


if __name__ == "__main__":
    main()
