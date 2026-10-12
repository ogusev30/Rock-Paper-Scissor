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

**Başlangıç modeli (karar: ben): Model B — ConvNeXt-Tiny.** Model A, Model B'yi kamerada denedikten sonra eğitilecek.

### Model A — Kendi CNN'im (sıfırdan) · onaylanan mimari
```
5 blok:  Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2   (kanallar 32 → 64 → 128 → 128 → 256, 224 px → 7×7)
Karar:   Flatten (12544) → Dropout 0.5 → Linear 2000 → ReLU → Dropout 0.5 → Linear 3
Toplam:  25,6M parametre (konvolüsyon blokları 0,54M; geri kalanı büyük Linear katmanı)
```
- Ağırlıklar rastgele başlar; AdamW (lr 1e-3, weight decay 1e-4), 60 epoch, cosine lr, label smoothing 0.1.
- Not: parametrelerin ~%98'i Linear katmanında → az veriyle ezber riski yüksek; Dropout + augmentation bu yüzden önemli.

### Model B — ConvNeXt-Tiny (pretrained) · CPU fine-tuning planı
- `timm` `convnext_tiny.fb_in22k_ft_in1k` (ImageNet-22k → 1k), 27,8M parametre, giriş 224 px.
- Adım 1: gövde donuk, sadece head (3,8K parametre) → 5 epoch, lr 1e-3.
- Adım 2: son aşama (stage 4) + head çözülür (15,5M parametre) → 10 epoch, lr 1e-4.
- `torch.set_num_threads(48)`; deneme çalıştırmasında epoch başına ~5-15 sn.

### Adil karşılaştırma
Aynı `dataset_split/` bölünmesi, aynı augmentation (RandomResizedCrop 0.75-1, ayna, ±20° döndürme, ColorJitter, %20 GaussianBlur), aynı ImageNet normalizasyonu, aynı batch (32), aynı seed (42). Fark sadece ön eğitim.

### Tahminim
**B az farkla kazanır** (macro F1'de ~0.05 civarı fark).

### Canlı panel
`exposure-academy-animations/animations/live.py` ve `live_cnn.html` değiştirmeden kopyalandı (birebir aynı oldukları `cmp` ile kontrol edildi).

---

## Aşama 4 — Training ve Metrik Karşılaştırma

### Model B — ConvNeXt-Tiny (pretrained) · 10 Ekim 2026
- Eğitim süresi: 1,3 dk (15 epoch, CPU). En iyi validation macro F1: **0.987** (epoch 6).
- Adım 1 (gövde donuk) sonunda val F1 0.930; adım 2'de son aşama çözülünce 0.987'ye çıktı.
- Train F1 1.000, val F1 0.987: belirgin bir ezber (overfitting) işareti yok; val loss eğitim boyunca düştü.

| Metrik (test, 73 foto) | Kağıt | Taş | Makas | **Macro F1** |
|---|---|---|---|---|
| Model B | 1.000 | 1.000 | 1.000 | **1.000** |

- ONNX: `model_b_fp32.onnx` (111 MB, PyTorch ile fark 3e-6, test F1 1.000) deneme sayfasında; `model_b.onnx` (int8, 34 MB, test F1 0.986) repo / web için.
- Not: Test datası küçük (73 foto) ve aynı kişi / aynı ortamlardan; 1.000 "mükemmel model" değil, "bu fotoğraflarda hiç hata yapmadı" demek. Asıl test kamerada.

### Model A — Kendi CNN'im (sıfırdan) · 12 Ekim 2026
- Eğitim süresi: 3,9 dk (60 epoch, CPU, epoch başına ~4 sn). En iyi validation macro F1: **0.986** (epoch 38).
- İlk epoch'larda F1 ~0.33 (rastgele tahmin); epoch 10'da val 0.60, epoch 20'de 0.85, epoch 30'da 0.96. Model B ilk epoch'ta 0.87'deydi → pretraining öğrenmeyi çok hızlandırıyor.
- Son epoch'ta train F1 0.976, val F1 0.971: ezber işareti yok (Dropout + augmentation işe yaradı).

| Metrik (test, 73 foto) | Kağıt | Taş | Makas | **Macro F1** |
|---|---|---|---|---|
| Model A | 1.000 | 0.981 | 0.979 | **0.987** |
| Model B | 1.000 | 1.000 | 1.000 | **1.000** |

- Tek hata: 1 taş fotoğrafı makas sanıldı.
- Pretraining farkı (B − A): **+0.013 macro F1**. Tahminim "B az farkla kazanır (~0.05)" idi → B kazandı, fark tahminimden de küçük.
- ONNX: `model_a_fp32.onnx` (103 MB, sadece sunucuda) ve `model_a.onnx` (int8, 27 MB, test F1 0.987; sıkıştırma Model A'da hiç F1 kaybettirmedi).
- Notebook düzeltmesi: `count` fonksiyonu Model B bölümünden ortak fonksiyonlar hücresine taşındı (Model A tek başına çalışmıyordu).

---

## Aşama 5 — Dene ve Karar Ver

---

## Aşama 6 — Productize ve Deploy
