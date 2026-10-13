"""Deneme sayfasında kaydedilen kameradan kareleri (dataset_live/) ham fotoğraflara (dataset/) taşır.

Kullanım:
    .venv/bin/python tools/import_live.py

Sonra her zamanki akış:
    .venv/bin/python tools/crop_hands.py      # elleri kırp (yeni kareler dahil)
    .venv/bin/python tools/split_dataset.py   # eski dosyalar yerinde kalır, sadece yeniler dağıtılır
    .venv/bin/python tools/crop_context.py    # bağlam kareleri (improve.py'nin yeni tarifi için)

Kayıtlı kareler zaten elin etrafından kesilmiş geniş karelerdir (el kutusu × 2.4); crop_hands.py elini
yeniden bulur ve aynı kuralla kırpar. Arka arkaya kaydedilmiş benzer kareleri split_dataset.py aynı
gruba koyar, böylece aynı hareketin neredeyse aynı kareleri hem train'e hem test'e düşmez.
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE, DATASET = ROOT / "dataset_live", ROOT / "dataset"
LABELS = ("rock", "paper", "scissors")


def main():
    for label in LABELS:
        files = sorted((LIVE / label).glob("*.jpg")) if (LIVE / label).exists() else []
        for f in files:
            shutil.move(str(f), DATASET / label / f.name)
        print(f"{label:<10}{len(files):>5} kare taşındı")


if __name__ == "__main__":
    main()
