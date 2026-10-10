"""iPhone HEIC fotoğraflarını JPEG'e çevirir.

Kullanım:
    .venv/bin/python tools/convert_heic.py

dataset/<sınıf>/ içindeki her .heic dosyasını aynı klasöre <sınıf>_iphone_<ad>.jpg
olarak kaydeder (döndürme bilgisi uygulanır, uzun kenar en fazla 1600 px).
Orijinal HEIC dosyası dataset_archive/heic/<sınıf>/ klasörüne taşınır.
Birebir aynı kopyalar (ör. Mac'in "IMG_1234 2.HEIC" kopyaları) çevrilmez,
dataset_archive/duplicates/<sınıf>/ klasörüne taşınır.
"""

import hashlib
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


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    total = dupes = 0
    for label in LABELS:
        folder = DATASET / label
        if not folder.exists():
            continue
        archived = ARCHIVE / label
        seen = {md5(p) for p in archived.iterdir()} if archived.exists() else set()
        for src in sorted(p for p in folder.iterdir() if p.suffix.lower() in {".heic", ".heif"}):
            digest = md5(src)
            if digest in seen:
                dup_dir = ROOT / "dataset_archive" / "duplicates" / label
                dup_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), dup_dir / src.name)
                print(f"KOPYA, atlandı: {label}/{src.name}")
                dupes += 1
                continue
            seen.add(digest)
            dst = folder / f"{label}_iphone_{src.stem.replace(' ', '_')}.jpg"
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
    print(f"\n{total} HEIC dosyası JPEG'e çevrildi, {dupes} kopya atlandı.")


if __name__ == "__main__":
    main()
