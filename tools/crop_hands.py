"""Aşama 2: MediaPipe ile her fotoğraftaki eli bulup kare olarak kırpar.

Kullanım:
    .venv/bin/python tools/crop_hands.py

Girdi:  dataset/<sınıf>/*.jpg
Çıktı:  dataset_cropped/<sınıf>/<ad>.jpg   (224x224, sadece el)
        dataset_no_hand/<sınıf>/<ad>.jpg   (el bulunamayan orijinaller)
        dataset_cropped/crops.csv           (her fotoğraf için el boyutu, bulanıklık vb.)

Kırpma kuralı (deneme sayfası ve web uygulaması da AYNI kuralı kullanmalı):
    21 el noktasının çevreleyen kutusu -> uzun kenarın 1.4 katı kare -> 224x224.
    Kare görüntünün dışına taşarsa taşan kısım siyah doldurulur.
"""

import csv
import ctypes
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# MediaPipe libEGL.so.1 ister; sunucuda yoksa tools/setup_gl.sh ile vendor/gl/ içine indirilir
for _lib in ("libGLdispatch.so.0", "libEGL.so.1", "libGLESv2.so.2"):
    if (ROOT / "vendor" / "gl" / _lib).exists():
        ctypes.CDLL(str(ROOT / "vendor" / "gl" / _lib), mode=ctypes.RTLD_GLOBAL)

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from PIL import Image, ImageOps

DATASET = ROOT / "dataset"
OUT = ROOT / "dataset_cropped"
NO_HAND = ROOT / "dataset_no_hand"
MODEL = ROOT / "models" / "hand_landmarker.task"
LABELS = ("rock", "paper", "scissors")
SIZE = 224
PAD = 1.4

_landmarkers = None


def _init():
    global _landmarkers
    cv2.setNumThreads(1)
    # Önce normal eşik, el bulunamazsa daha düşük eşikle bir kez daha dene
    _landmarkers = [
        HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL)),
            running_mode=RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=conf, min_hand_presence_confidence=conf))
        for conf in (0.5, 0.2)
    ]


def square_box(landmarks, w, h):
    xs = np.array([p.x for p in landmarks]) * w
    ys = np.array([p.y for p in landmarks]) * h
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * PAD
    cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
    return cx - side / 2, cy - side / 2, side


def process(path):
    label = path.parent.name
    with Image.open(path) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    h, w = rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    result, pass_no = None, 0
    for pass_no, lm in enumerate(_landmarkers, 1):
        result = lm.detect(mp_image)
        if result.hand_landmarks:
            break
    row = {"file": path.name, "label": label, "width": w, "height": h}
    if not result.hand_landmarks:
        dst = NO_HAND / label / path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        return {**row, "found": 0}

    # Birden fazla el varsa en büyüğünü al
    boxes = [square_box(l, w, h) for l in result.hand_landmarks]
    best = int(np.argmax([b[2] for b in boxes]))
    x0, y0, side = boxes[best]
    crop = Image.fromarray(rgb).crop((round(x0), round(y0), round(x0 + side), round(y0 + side)))
    crop = crop.resize((SIZE, SIZE), Image.LANCZOS)
    dst = OUT / label / path.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    crop.save(dst, "JPEG", quality=95)

    gray = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2GRAY)
    return {**row, "found": 1, "pass": pass_no, "n_hands": len(boxes),
            "score": round(result.handedness[best][0].score, 3),
            "hand": result.handedness[best][0].category_name,
            "crop_px": round(side), "crop_frac": round(side / max(w, h), 3),
            "blur": round(cv2.Laplacian(gray, cv2.CV_64F).var(), 1)}


def main():
    for d in (OUT, NO_HAND):
        if d.exists():
            shutil.rmtree(d)
    files = [p for label in LABELS for p in sorted((DATASET / label).glob("*.jpg"))]
    with Pool(min(os.cpu_count(), 48), initializer=_init) as pool:
        rows = pool.map(process, files, chunksize=4)

    fields = ["file", "label", "found", "pass", "n_hands", "score", "hand",
              "crop_px", "crop_frac", "blur", "width", "height"]
    OUT.mkdir(exist_ok=True)
    with open(OUT / "crops.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'Sınıf':<10}{'Toplam':>8}{'Kırpıldı':>10}{'El yok':>8}")
    for label in LABELS:
        r = [x for x in rows if x["label"] == label]
        found = sum(x["found"] for x in r)
        print(f"{label:<10}{len(r):>8}{found:>10}{len(r) - found:>8}")


if __name__ == "__main__":
    main()
