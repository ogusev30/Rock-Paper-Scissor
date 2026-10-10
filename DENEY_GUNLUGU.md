# Deney Günlüğü — Rock Paper Scissors Vision

Bu dosya projenin her aşamasındaki kararları, sayıları ve gözlemleri içerir.

---

## Aşama 1 — Dataset Toplama

**Hedef:** Her sınıf için en az 150, toplamda en az 450 fotoğraf.

### Çekim planı (her hareket için)

| Neyi değiştiriyorum? | Nasıl? |
|---|---|
| Arka plan | Düz duvar, masa üstü, karışık oda — her birinde ~50 fotoğraf |
| Açı | Önden, sağa eğik, sola eğik, hafif yukarıdan, hafif aşağıdan, avuç içi, el üstü |
| Mesafe | Yakın (el kadrajın yarısı), orta, uzak (el kadrajın dörtte biri) |
| El | Hem sağ hem sol el |
| Cihaz | Yarısı telefon, yarısı laptop kamerası |

### Araçlar

- Çekim sayfası: `python tools/capture/server.py` → https://student.exposureai.org/lab/proxy/8000/
- Sayım: `python tools/count_dataset.py`

### Sınıf sayıları

| Sınıf | Fotoğraf sayısı |
|---|---|
| rock (taş) | — |
| paper (kağıt) | — |
| scissors (makas) | — |
| **Toplam** | — |

### Notlar

- iPhone HEIC fotoğrafları `tools/convert_heic.py` ile, WhatsApp zip'leri `tools/import_zip.py` ile JPEG'e çevrildi.
- Kopya temizliği: birebir aynı dosyalar (md5) ve WhatsApp'ın sıkıştırdığı kopyalar (görsel benzerlik / dHash) datasetten çıkarıldı → `dataset_archive/duplicates/`. Taş: 18 kopya, kağıt: 35 kopya (aynı fotoğraflar hem iPhone hem WhatsApp'tan gelmişti, kaliteli iPhone hali tutuldu).
- Yanlış klasör: kağıt olarak gelen 50 WhatsApp fotoğrafı taştan kağıda, kağıt klasöründeki 45 yumruk fotoğrafı kağıttan taşa taşındı.
- Belirsiz 2 fotoğraf (yarı açık el, kadrajdan taşan el) hiçbir sınıfa konmadı → `dataset_review/`.

---

## Aşama 2 — Data Cleaning

---

## Aşama 3 — Model Seçimi

---

## Aşama 4 — Training ve Metrik Karşılaştırma

---

## Aşama 5 — Dene ve Karar Ver

---

## Aşama 6 — Productize ve Deploy
