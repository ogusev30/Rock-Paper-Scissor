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
| rock (taş) | 195 |
| paper (kağıt) | 150 |
| scissors (makas) | 155 |
| **Toplam** | **500** |

(10 Ekim 2026 itibarıyla, kopyalar ve yanlış klasördekiler temizlendikten sonra.)

### Notlar

- iPhone HEIC fotoğrafları `tools/convert_heic.py` ile, WhatsApp zip'leri `tools/import_zip.py` ile JPEG'e çevrildi.
- Kopya temizliği: birebir aynı dosyalar (md5) ve WhatsApp'ın sıkıştırdığı kopyalar (görsel benzerlik / dHash) datasetten çıkarıldı → `dataset_archive/duplicates/`. Taş: 18 kopya, kağıt: 35 kopya (aynı fotoğraflar hem iPhone hem WhatsApp'tan gelmişti, kaliteli iPhone hali tutuldu).
- Yanlış klasör: kağıt olarak gelen 50 WhatsApp fotoğrafı taştan kağıda, kağıt klasöründeki 45 yumruk fotoğrafı kağıttan taşa taşındı.
- Belirsiz 2 fotoğraf (yarı açık el, kadrajdan taşan el) hiçbir sınıfa konmadı → `dataset_review/`.

---

## Aşama 2 — Data Cleaning

### Eli kırpma
- `tools/crop_hands.py`: MediaPipe Hand Landmarker (float16) ile 21 el noktası bulunur; noktaların kutusunun uzun kenarı × 1.4 kare olarak kırpılır, 224×224'e boyutlanır. Kare görüntü dışına taşarsa siyah doldurulur. **Deneme sayfası ve web uygulaması da aynı kuralı kullanacak.**
- Önce eşik 0.5, el bulunamazsa 0.2 ile tekrar denenir. 500 fotoğrafın 487'sinde el bulundu (18'i ikinci denemede).
- Not: MediaPipe 1.x sunucuda `libEGL.so.1` istiyor; root olmadığı için `tools/setup_gl.sh` ile Debian paketlerinden `vendor/gl/` içine indirildi (sunucu ARM / aarch64).

### Silinenler (karar: ben) → `dataset_archive/removed/`
| Neden | Taş | Kağıt | Makas |
|---|---|---|---|
| El bulunamadı (çok yakın/kesik, çok uzak, MediaPipe kaçırdı) | 6 | 1 | 6 |
| Bulanık (hareket / çok uzaktan webcam) | 11 | 0 | 0 |

- Kapalı makas (iki parmak bitişik, ~18 foto) **tutuldu**: taş/kağıtla karışmaz ve demoda insanlar böyle de yapacak.

### Bölme
- `tools/split_dataset.py`: sınıf başına ~%70 / %15 / %15, seed 42. Arka arkaya çekilmiş neredeyse aynı kareler (dHash ≤ 12) aynı gruba konup birlikte tek bölüme gönderildi (19 grup), böylece test datasında eğitimdekinin neredeyse aynısı yok.

| Sınıf | train | val | test | toplam |
|---|---|---|---|---|
| rock | 124 | 27 | 27 | 178 |
| paper | 105 | 23 | 23 | 151 |
| scissors | 107 | 23 | 23 | 153 |
| **toplam** | **336** | **73** | **73** | **482** |

- Temizlik sonrası kağıt/makas 149'a düşmüştü; 2 kağıt + 4 makas daha çekildi, script'ler yeniden çalıştırıldı.

---

## Aşama 3 — Model Seçimi

---

## Aşama 4 — Training ve Metrik Karşılaştırma

---

## Aşama 5 — Dene ve Karar Ver

---

## Aşama 6 — Productize ve Deploy
