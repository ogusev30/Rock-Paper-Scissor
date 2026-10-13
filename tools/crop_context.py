"""İyileştirme: her eli, normal kırpmadan daha geniş bir "bağlam" karesi olarak kırpar.

Kullanım:
    .venv/bin/python tools/crop_context.py

Girdi:  dataset_split/split.csv + dataset/<sınıf>/*.jpg (ham fotoğraflar)
Çıktı:  dataset_context/<bölüm>/<sınıf>/<ad>.jpg   (384x384)

Kare, crop_hands.py ile aynı el kutusunun merkezinde ama uzun kenarın CTX (2.4) katı genişliğinde.
Ortasındaki 1.4 / 2.4'lük parça normal kırpmanın aynısıdır. Eğitimde bu karenin içinden rastgele pay
(1.15–1.75) ve kaymayla kesilir: kamerada MediaPipe'ın kutusu her karede biraz oynar, model buna alışır.
Taşan kısım siyah (crop_hands.py ile aynı kural).
"""

import csv
import os
import shutil
from multiprocessing import Pool

import numpy as np

import crop_hands as ch  # GL kütüphanelerini yükler, MediaPipe ayarlarını paylaşır
from PIL import Image, ImageOps

CTX = 2.4
SIZE = 384
OUT = ch.ROOT / "dataset_context"


def process(row):
    src = ch.DATASET / row["label"] / row["file"]
    with Image.open(src) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    h, w = rgb.shape[:2]
    mp_image = ch.mp.Image(image_format=ch.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = None
    for lm in ch._landmarkers:
        result = lm.detect(mp_image)
        if result.hand_landmarks:
            break
    if not result.hand_landmarks:
        return row["file"], False
    boxes = [ch.square_box(l, w, h) for l in result.hand_landmarks]
    x0, y0, side = boxes[int(np.argmax([b[2] for b in boxes]))]
    cx, cy, big = x0 + side / 2, y0 + side / 2, side / ch.PAD * CTX
    crop = Image.fromarray(rgb).crop((round(cx - big / 2), round(cy - big / 2),
                                      round(cx + big / 2), round(cy + big / 2)))
    dst = OUT / row["split"] / row["label"] / row["file"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    crop.resize((SIZE, SIZE), Image.LANCZOS).save(dst, "JPEG", quality=95)
    return row["file"], True


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    rows = list(csv.DictReader(open(ch.ROOT / "dataset_split" / "split.csv")))
    with Pool(min(os.cpu_count(), 48), initializer=ch._init) as pool:
        done = pool.map(process, rows, chunksize=4)
    missing = [f for f, ok in done if not ok]
    print(f"{len(done) - len(missing)} / {len(done)} bağlam karesi yazıldı" + (f", el bulunamadı: {missing}" if missing else ""))


if __name__ == "__main__":
    main()
