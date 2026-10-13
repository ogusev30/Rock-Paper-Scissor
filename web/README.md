# Taş Kağıt Makas AI — web uygulaması (Aşama 6)

Elini göster, ne yaptığını tahmin edeyim. Her şey tarayıcıda çalışır: MediaPipe Hand Landmarker eli bulur,
eğitilen model (ONNX Runtime Web) kırpılmış ele bakıp taş / kağıt / makas der. Sunucu ve API anahtarı yok.

## Çalıştırma

```bash
cd web
npm install
npm run dev        # http://localhost:3000
```

`npm run dev` ve `npm run build` önce `scripts/copy-assets.mjs`'i çalıştırır: ONNX Runtime Web ve MediaPipe
dosyaları `node_modules`'tan `public/` altına kopyalanır (CDN'e bağımlılık yok).

## Model

`public/models/`:
- `model.json`: modelin adı, test macro F1, sınıflar, normalizasyon, kırpma payı (`improve.py export` yazar)
- `model_*.onnx`: int8 model (web sayfası `model.json`'daki `file` adını yükler)
- `hand_landmarker.task`: MediaPipe el tespit modeli

Kırpma kuralı eğitimdekiyle aynı (`tools/crop_hands.py`): 21 el noktasının kutusu → uzun kenar × 1.4 kare → 224×224.

## Deploy (Vercel)

Vercel'de projeyi GitHub reposundan içe aktar, **Root Directory = `web`** seç. Framework otomatik Next.js
olarak algılanır; environment variable gerekmez. `next.config.mjs` COOP/COEP başlıklarını ekler, böylece
ONNX Runtime Web birden çok çekirdekle çalışır.
