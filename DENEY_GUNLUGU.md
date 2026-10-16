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

### Yeni fotoğraflar · 13 Ekim 2026
- Kamerada model makası karıştırdığı için yeni iPhone fotoğrafları eklendi: kağıt +98, makas +78 (HEIC → JPEG, `tools/convert_heic.py`).
- Yeni ham sayılar: taş 178, kağıt 249, makas 231 (toplam 658).
- ⚠️ Cihaz dengesi: makasın 230 fotoğrafından sadece **5'i laptop kamerasından** (taş 19, kağıt 9). Plan "yarısı laptop" diyordu ve demo laptop kamerasıyla. Kamerada makas sorununun en olası sebebi bu → deneme sayfasından laptop kamerasıyla kare kaydedilecek (`dataset_live/`, `tools/import_live.py`).

### Dizüstü kamera kareleri ve son fotoğraflar · 16 Ekim 2026
- Deneme sayfasında kaydedilen 89 dizüstü kamera karesi (taş 8, kağıt 52, makas 29) temizlenip datasete eklendi.
- Sonra 46 WhatsApp fotoğrafı daha eklendi (taş 12, kağıt 17, makas 17).

| Sınıf | Son ham sayı |
|---|---|
| rock (taş) | 198 |
| paper (kağıt) | 318 |
| scissors (makas) | 277 |
| **Toplam** | **793** |

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

**Yeni fotoğraflardan sonra (13 Ekim):** eski dosyalar bölümlerinde kaldı (test seti değişmedi, sadece büyüdü), yeniler oranları tutacak şekilde dağıtıldı. 1 makas fotoğrafında el bulunamadı.

| Sınıf | train | val | test | toplam |
|---|---|---|---|---|
| rock | 124 | 27 | 27 | 178 |
| paper | 173 | 38 | 38 | 249 |
| scissors | 160 | 35 | 35 | 230 |
| **toplam** | **457** | **100** | **100** | **657** |

**Son bölünme (16 Ekim, dizüstü kamera kareleri + yeni fotoğraflar):** 793 fotoğrafın 790'ında el bulundu. Eski dosyalar yine bölümlerinde kaldı.

| Sınıf | train | val | test | toplam |
|---|---|---|---|---|
| rock | 137 | 30 | 30 | 197 |
| paper | 222 | 48 | 48 | 318 |
| scissors | 192 | 42 | 41 | 275 |
| **toplam** | **551** | **120** | **119** | **790** |
| *bunların dizüstü kamera karesi* | *64* | *13* | *12* | *89* |

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

### V2 — Kameraya dayanıklı tarif + yeni fotoğraflar · 12–13 Ekim 2026
Kamerada iki model de ilk sürümde zorlandı (özellikle makas, kutu kayınca ve ışık değişince). Test seti dolu (B 1.000) olduğu için ölçüm yetmiyordu; bu yüzden:
- **Zor test** (`improve.py`): aynı val / test fotoğraflarının 12 bozulmuş hali (kutu kayması, kırpma payı, eğik el, karanlık, parlak, gürültü, hareket bulanıklığı, JPEG, düşük çözünürlük, siyah-beyaz). Tarif ve epoch seçimi sadece val ile; test sadece raporlandı.
- **Yeni tarif**: el, her seferinde geniş bağlam karesinden (`tools/crop_context.py`, el kutusu × 2.4) rastgele pay (1.2–1.7) ve kaymayla kesilir + kamera bozulmaları. İki model yine aynı data, aynı tarif, aynı bölünme.
- İlk datasetle (482 el) notebook modelleri zor testte: A 0.749, B 0.972. Yeni tarifle: A 0.959, B 0.996.
- Sonra yeni fotoğraflar eklendi (657 el) ve iki model **`training.ipynb` V2 bölümünde** yeniden eğitildi (B 8 dk, A 33 dk). Deneme sayfası artık bu modelleri kullanıyor.

| Test (100 foto, yeni data) | Kağıt | Taş | Makas | **Macro F1** | Zor test F1 |
|---|---|---|---|---|---|
| Model A — kendi CNN (`A_new2_long`) | 0.973 | 1.000 | 0.972 | **0.982** | 0.931 |
| Model B — ConvNeXt-Tiny (`B_deep`) | 1.000 | 1.000 | 1.000 | **1.000** | 0.988 |

- Pretraining farkı (B − A): temiz testte **+0.018**, zor testte **+0.057**. Fark asıl bozulmuş (kameraya benzer) görüntülerde ortaya çıkıyor.
- A en çok kutu kaymasında (sol-yukarı 0.804) ve eğik elde (0.905) düşüyor; B'nin en kötüsü düşük çözünürlük (0.971).
- int8 ONNX: A 27 MB, B 34 MB; ikisi de fp32 ile aynı test F1.
- Eski modeller `models/eski/`, eski sonuçlar `results/v2/eski_veri/` içinde.

### V3 — Dizüstü kamera kareleri + distillation denemesi · 16 Ekim 2026
Deneme sayfasında kaydedilen 89 dizüstü kamera karesinde (eğitimde hiç görülmemiş) V2 modelleri: **A 52/89 (%58), B 87/89 (%98)**. A kağıtların yarısından fazlasını makas sanıyordu. Sebep: eğitim fotoğraflarının neredeyse hepsi iPhone'dan.
- Kareler önce elle temizlendi: yanlış butonla kaydedilen 10 makas → `scissors`; emin olunamayan 11 kare ve yanlış etiketli 4 kare silindi.
- `tools/import_live.py` → crop → split (eski dosyalar yerinde, yeni gruplar dağıtıldı) → bağlam kareleri. Dataset 746 el (521 / 113 / 112).
- **Canlı test**: test bölümüne düşen 12 canlı kare (val'de 13). Asıl hedef bu küme; az olduğu için 1 kare ≈ 0.07 F1.
- Tarif **new3** = new2 + canlı kareler eğitimde 3 kat sık + renk kayması (farklı kameraların beyaz dengesi). Val'deki canlı kareler model seçiminde ayrı küme olarak sayılır.
- **A_kd**: Model A yine sıfırdan; kayıp = ½ etiket + ½ öğretmen B_live'ın olasılıkları (T = 2).

| (test: 112 foto, canlı 12) | Temiz F1 | **Canlı F1** | Canlı val F1 | Zor F1 | Süre |
|---|---|---|---|---|---|
| A V2 (`A_new2_long`, canlı kare yok) | 0.942 | 0.593 | 0.556 | 0.898 | – |
| B V2 (`B_deep`, canlı kare yok) | 0.992 | 0.937 | 0.886 | 0.982 | – |
| **A V3 (`A_live`)** | 0.950 | **0.930** | 0.941 | 0.929 | 42 dk |
| A V3 distillation (`A_kd`) | 0.950 | 1.000 | 0.941 | 0.926 | 59 dk |
| **B V3 (`B_live`)** | 1.000 | **1.000** | 1.000 | 0.983 | 10 dk |

- Canlı kareler A'yı kamerada **0.59 → 0.93** çıkardı; B − A farkı canlıda 0.34 → 0.07.
- A_kd val'de A_live ile neredeyse aynı (temiz 0.948 / 0.940, canlı ikisi 0.941, val loss biraz kötü). Canlı testteki fark tek kare. **Sayfaya A_live kondu**: distillation B'nin pretraining bilgisini A'ya taşıdığı için "pretraining farkı" karşılaştırmasını bozar ve val'de açık bir kazancı yok.
- Aynalama TTA'sı (görüntü + yatay ayna ortalaması) denendi: hiçbir modelde val'de kazanç yok (B'de canlı val 1.000 → 0.941) → kullanılmadı.
- int8 ONNX: A fp32 ile aynı (test 0.950); B test temiz 1.000 → 0.984 (2 foto), canlı 1.000 aynı. Eski V2 modelleri `models/eski/v2_s1_20261013/`.

### Son modeller — 790 el · 16 Ekim 2026 (`training.ipynb` V3 bölümü)
Yeni fotoğraflar eklenince iki model aynı tarifle (`B_live`, `A_live`) yeniden eğitildi. Test seti değiştiği için (119 foto) yukarıdaki tabloyla birebir karşılaştırılamaz.

| Test (119 foto, canlı 12) | Kağıt | Taş | Makas | **Macro F1** | Canlı F1 | Zor F1 | Süre |
|---|---|---|---|---|---|---|---|
| Model A — kendi CNN (`A_live`) | 0.925 | 0.966 | 0.920 | **0.937** | 1.000 | 0.911 | 42 dk |
| Model B — ConvNeXt-Tiny (`B_live`) | 1.000 | 1.000 | 1.000 | **1.000** | 1.000 | 0.989 | 10 dk |

- **Pretraining farkı (B − A): temiz +0.063, zor +0.078**; canlı testte ikisi eşit (12 kare az olduğu için bu kümede fark görünmüyor).
- A'nın en zayıf olduğu yer kutu kayması (0.84–0.86) ve eğik el (0.875); B'nin en kötüsü hareket bulanıklığı (0.959).
- Aşama 3'teki tahminim "B az farkla kazanır (~0.05)" idi → **B kazandı**; son modellerde fark tahmine çok yakın (+0.063), zor testte daha da büyük.
- int8 ONNX: B 34 MB, test F1 1.000 (fp32 ile aynı).

---

## Aşama 5 — Dene ve Karar Ver

### Deneme sayfası (`play.py`, https://student.exposureai.org/lab/proxy/8001/)
- İki model yan yana: bounding box, "Taş %94" etiketi, olasılık çubukları, "Elini kameraya göster" / "Kameraya erişilemedi. Lütfen izin ver." durumları.
- Gecikme düzeltmeleri: aynı anda 3 istek yolda olabilir, zaman yumuşatma (120 ms).
- **Hız (13 Ekim):** sunucu artık int8 modelleri paralel çalıştırıyor (model başına 16 çekirdek). Bir tahmin isteği 35 ms → 14 ms; B tek başına 28 → 12 ms, A ~2.5 ms. Sayfada her modelin süresi ⏱ ile yazıyor.
- **Sunucu / Tarayıcı** seçimi: "Tarayıcı"da modeller ONNX Runtime Web ile tarayıcıda çalışır (proxy gidiş-dönüşü yok). Sunucuda tek çekirdekli WASM ölçümü: A 66 ms, B ~440 ms. B büyük model olduğu için tarayıcıda yavaş; laptopta çok çekirdekle daha hızlı olmalı (Aşama 6'da COOP/COEP ile açık).
- Kamerada yanlış tahmin olunca 1 / 2 / 3 tuşlarıyla kare kaydedilir → `dataset_live/` → `tools/import_live.py` ile datasete eklenir.

### Karar (16 Ekim 2026) — web'e **Model B (ConvNeXt-Tiny)** gidiyor
- **Neden:** son modelleri deneme sayfasında yan yana denedim; **B kamerada daha tutarlıydı.**
- **Metrikler ve canlı deneme aynı şeyi söyledi mi?** Evet. B temiz testte (1.000 / 0.937) ve zor testte (0.989 / 0.911) önde; canlı testte ikisi eşit (1.000), kamerada ise B daha tutarlıydı.
- **Model A nerede zorlandı?** Kutu kayınca ve el eğikken (zor testte 0.84–0.88); ilk sürümlerde laptop kamerasında kağıdı makas sanıyordu. A'nın artısı hız (tarayıcıda ~7 kat hızlı).

---

## Aşama 6 — Productize ve Deploy

### Taslak (13 Ekim) — `web/`
- Next.js 15 + ONNX Runtime Web 1.20 (WASM) + MediaPipe Tasks Vision; her şey tarayıcıda, sunucu / API anahtarı yok.
- Tek model, büyük kamera görüntüsü, bounding box + % confidence, tahmin kartı, model bilgi kartı (ad, test macro F1, tahmin süresi), yükleniyor / hata / "elini göster" durumları, ön kamera, mobil uyumlu yerleşim.
- ORT ve MediaPipe dosyaları `public/` altından gelir (CDN yok); COOP/COEP başlıklarıyla WASM çok çekirdekli çalışır.

### Son ürün (16 Ekim) — double down: Model B
- **Ürün adı:** Taş Kağıt Makas AI — *Elini göster, ne yaptığını tahmin edeyim.*
- Web uygulamasında sadece kazanan model var: `web/public/models/model_b.onnx` (son B, test macro F1 1.000), model kartında adı ve macro F1'i yazıyor.
- **Kaybeden Model A kaldırıldı:** `models/model_a.onnx` / `model_a.json` repodan silindi; web uygulamasında A'ya ait kod hiç yoktu. Eğitim kodu (`improve.py`, `training.ipynb`) karşılaştırma için kaldı.
- Favicon eklendi (`web/app/icon.svg`, `apple-icon.png`): makas yapan el.
- Yapılacaklar (öğrenci): telefonda ön kamerayla deneme, Vercel deploy (Root Directory = `web`), canlı linki telefondan açmak.
