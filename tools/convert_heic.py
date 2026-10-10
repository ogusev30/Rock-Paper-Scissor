"""iPhone HEIC fotoğraflarını JPEG'e çevirir.

Kullanım:
    .venv/bin/python tools/convert_heic.py

dataset/<sınıf>/ içindeki her .heic dosyasını aynı klasöre <sınıf>_iphone_<ad>.jpg
olarak kaydeder (döndürme bilgisi uygulanır, uzun kenar en fazla 1600 px).
Orijinal HEIC dosyası dataset_archive/heic/<sınıf>/ klasörüne taşınır.
"""

import shutil
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
ARCHIVE = ROOT / "dataset_archive" / "heic"
LABELS = ("rock", "paper", "scissors")
MAX_SIDE = 1600


def main():
    total = 0
    for label in LABELS:
        folder = DATASET / label
        if not folder.exists():
            continue
        for src in sorted(p for p in folder.iterdir() if p.suffix.lower() in {".heic", ".heif"}):
            dst = folder / f"{label}_iphone_{src.stem}.jpg"
            try:
                with Image.open(src) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    im.thumbnail((MAX_SIDE, MAX_SIDE))
                    im.save(dst, "JPEG", quality=92)
            except Exception as e:
                print(f"ATLANDI: {src.name} ({e})")
                continue
            (ARCHIVE / label).mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), ARCHIVE / label / src.name)
            print(f"{label}/{src.name} -> {dst.name} {im.size}")
            total += 1
    print(f"\n{total} HEIC dosyası JPEG'e çevrildi.")


if __name__ == "__main__":
    main()
