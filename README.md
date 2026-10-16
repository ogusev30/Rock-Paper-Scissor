# Taş Kağıt Makas AI

Elini göster, ne yaptığını tahmin edeyim. Laptop ya da telefon kamerasında taş, kağıt ve makas hareketlerini anlık tanıyan bir web uygulaması.
Exposure AI Academy — Proje 1 (Rock Paper Scissors Vision).

- **Eli bulmak:** MediaPipe Hand Landmarker
- **Hareketi tanımak:** kendi fotoğraflarımla fine-tune edilmiş ConvNeXt-Tiny (int8 ONNX, test macro F1 1.000)
- Her şey tarayıcıda çalışır (ONNX Runtime Web), sunucu veya API anahtarı yok.

## Repoda neler var?

| Yol | Ne? |
|---|---|
| `training.ipynb` | Bütün eğitim akışı: dataset, augmentation, Model A (sıfırdan CNN) ve Model B (ConvNeXt-Tiny), metrikler, Grad-CAM, karşılaştırma, ONNX |
| `DENEY_GUNLUGU.md` | Her aşamanın kararları, sayıları ve gözlemleri |
| `improve.py` | Kameraya dayanıklı eğitim tarifleri, zor test ve canlı test (notebook'un V2 / V3 bölümleri bunu çağırır) |
| `models/model_b.onnx` | Kazanan model (int8) |
| `web/` | Son ürün: Next.js + ONNX Runtime Web — ayrıntılar `web/README.md` |
| `play.py`, `play.html` | Aşama 5 deneme sayfası (modelleri yan yana deneme) |
| `tools/` | Fotoğraf çekme sayfası, HEIC/zip dönüştürme, el kırpma, bölme |
| `live.py`, `live_cnn.html` | Canlı eğitim paneli ([exposure-academy-animations](https://github.com/Darkosxl/exposure-academy-animations), değiştirilmeden) |

Ham fotoğraflar (`dataset/`, `dataset_*/`) kişisel olduğu için repoda değil.

## Çalıştırma

```bash
cd web
npm install
npm run dev        # http://localhost:3000
```

Vercel: projeyi bu repodan içe aktar, **Root Directory = `web`**. Environment variable gerekmez.
