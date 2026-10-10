"""Her sınıftaki fotoğraf sayısını gösterir (Aşama 1 kontrol noktası).

Kullanım:
    python tools/count_dataset.py
"""

from pathlib import Path

DATASET = Path(__file__).resolve().parents[1] / "dataset"
LABELS = {"rock": "Taş", "paper": "Kağıt", "scissors": "Makas"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
TARGET = 150


def main():
    counts = {}
    for label in LABELS:
        folder = DATASET / label
        counts[label] = sum(1 for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS) if folder.exists() else 0

    print(f"{'Sınıf':<18}{'Sayı':>6}  Durum")
    print("-" * 36)
    for label, name in LABELS.items():
        n = counts[label]
        status = "tamam" if n >= TARGET else f"{TARGET - n} eksik"
        print(f"{name + ' (' + label + ')':<18}{n:>6}  {status}")
    print("-" * 36)
    print(f"{'Toplam':<18}{sum(counts.values()):>6}")

    low, high = min(counts.values()), max(counts.values())
    if high and low < 0.8 * high:
        print(f"\nUyarı: sınıflar dengesiz (en az {low}, en çok {high}). Sayıları birbirine yaklaştır.")
    if low >= TARGET and low >= 0.8 * high:
        print("\nKontrol noktası geçti: her sınıfta en az 150 fotoğraf var ve sayılar dengeli.")

    heic = [p for p in DATASET.rglob("*") if p.suffix.lower() == ".heic"]
    if heic:
        print(f"\nNot: {len(heic)} HEIC dosyası var (iPhone). Aşama 2'de JPEG'e çevrilecek.")


if __name__ == "__main__":
    main()
