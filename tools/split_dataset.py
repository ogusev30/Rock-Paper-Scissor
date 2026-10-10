"""Aşama 2: Kırpılmış eli train / validation / test olarak böler (~%70 / %15 / %15).

Kullanım:
    .venv/bin/python tools/split_dataset.py

Girdi:  dataset_cropped/<sınıf>/*.jpg
Çıktı:  dataset_split/{train,val,test}/<sınıf>/*.jpg
        dataset_split/split.csv   (her dosyanın hangi bölüme gittiği)

Arka arkaya çekilmiş, neredeyse aynı kareler (dHash farkı <= 12) aynı gruba konur ve
grup bütün olarak tek bir bölüme gider. Böylece test datasında, eğitimde görülmüş bir
fotoğrafın neredeyse aynısı olmaz.
"""

import csv
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "dataset_cropped"
OUT = ROOT / "dataset_split"
LABELS = ("rock", "paper", "scissors")
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
NEAR_DUP = 12
SEED = 42


def dhash(path, n=16):
    a = np.asarray(Image.open(path).convert("L").resize((n + 1, n), Image.LANCZOS), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def groups_of(files):
    """Neredeyse aynı kareleri birleştirir (union-find)."""
    hashes = np.array([dhash(f) for f in files])
    dist = (hashes[:, None] != hashes[None]).sum(-1)
    parent = list(range(len(files)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.where(np.triu(dist <= NEAR_DUP, k=1))):
        parent[find(i)] = find(j)
    groups = {}
    for i, f in enumerate(files):
        groups.setdefault(find(i), []).append(f)
    return list(groups.values())


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    rng = random.Random(SEED)
    rows = []
    for label in LABELS:
        files = sorted((SRC / label).glob("*.jpg"))
        groups = groups_of(files)
        rng.shuffle(groups)
        # Büyük grupları önce yerleştir, her grubu hedefine en uzak bölüme koy
        groups.sort(key=len, reverse=True)
        target = {s: r * len(files) for s, r in RATIOS.items()}
        filled = {s: 0 for s in RATIOS}
        for g in groups:
            split = max(RATIOS, key=lambda s: (target[s] - filled[s]) / target[s])
            filled[split] += len(g)
            for f in g:
                dst = OUT / split / label / f.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                rows.append({"file": f.name, "label": label, "split": split, "group": len(g)})

    with open(OUT / "split.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", "split", "group"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'Sınıf':<10}{'train':>7}{'val':>6}{'test':>6}{'toplam':>8}")
    for label in LABELS + ("toplam",):
        r = [x for x in rows if label in (x["label"], "toplam")]
        counts = [sum(x["split"] == s for x in r) for s in RATIOS]
        print(f"{label:<10}{counts[0]:>7}{counts[1]:>6}{counts[2]:>6}{len(r):>8}")


if __name__ == "__main__":
    main()
