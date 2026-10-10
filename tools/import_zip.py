"""dataset/<sınıf>/ içine atılan zip dosyalarındaki fotoğrafları açar.

Kullanım:
    .venv/bin/python tools/import_zip.py

Her zip'teki resimler kontrol edilir (gerçekten resim mi?) ve aynı klasöre
<sınıf>_whatsapp_<zip saati>_<sıra>.jpg olarak kaydedilir. Klasörde birebir aynısı
zaten olan fotoğraflar atlanır. Zip dosyası dataset_archive/zips/<sınıf>/ klasörüne taşınır.
"""

import hashlib
import io
import re
import shutil
import zipfile
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
ARCHIVE = ROOT / "dataset_archive" / "zips"
LABELS = ("rock", "paper", "scissors")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
MAX_SIDE = 1600
MAX_FILE = 30 * 1024 * 1024


def md5(data):
    return hashlib.md5(data).hexdigest()


def zip_tag(zip_path):
    # "WhatsApp Unknown 2026-10-10 at 6.35.11 PM.zip" -> "20261010_183511"
    m = re.search(r"(\d{4})-(\d{2})-(\d{2}) at (\d{1,2})\.(\d{2})\.(\d{2}) ?(AM|PM)?", zip_path.name)
    if not m:
        return md5(zip_path.name.encode())[:8]
    y, mo, d, h, mi, s, ampm = m.groups()
    h = int(h) % 12 + (12 if ampm == "PM" else 0) if ampm else int(h)
    return f"{y}{mo}{d}_{h:02d}{mi}{s}"


def main():
    for label in LABELS:
        folder = DATASET / label
        if not folder.exists():
            continue
        zips = sorted(folder.glob("*.zip"))
        if not zips:
            continue
        seen = {md5(p.read_bytes()) for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS}
        for zip_path in zips:
            tag = zip_tag(zip_path)
            saved = dupes = skipped = 0
            with zipfile.ZipFile(zip_path) as zf:
                entries = [i for i in zf.infolist()
                           if not i.is_dir() and Path(i.filename).suffix.lower() in IMAGE_EXTS
                           and not Path(i.filename).name.startswith("._")]
                for n, info in enumerate(entries, 1):
                    if info.file_size > MAX_FILE:
                        skipped += 1
                        continue
                    data = zf.read(info)
                    if md5(data) in seen:
                        dupes += 1
                        continue
                    try:
                        with Image.open(io.BytesIO(data)) as im:
                            im = ImageOps.exif_transpose(im).convert("RGB")
                            im.thumbnail((MAX_SIDE, MAX_SIDE))
                            dst = folder / f"{label}_whatsapp_{tag}_{n:03d}.jpg"
                            im.save(dst, "JPEG", quality=95)
                    except Exception as e:
                        print(f"ATLANDI (resim değil): {info.filename} ({e})")
                        skipped += 1
                        continue
                    seen.add(md5(data))
                    seen.add(md5(dst.read_bytes()))
                    saved += 1
            (ARCHIVE / label).mkdir(parents=True, exist_ok=True)
            shutil.move(str(zip_path), ARCHIVE / label / zip_path.name)
            print(f"{label}/{zip_path.name}: {saved} fotoğraf eklendi, {dupes} kopya atlandı, {skipped} atlandı")


if __name__ == "__main__":
    main()
