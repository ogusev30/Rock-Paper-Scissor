"""Aşama 5: Deneme sayfası — eğitilen modelleri kendi elinle kamerada dene.

Kullanım:
    .venv/bin/python play.py            # terminalde yazan linki aç

Tarayıcı: kamerayı açar, MediaPipe ile eli bulur ve eğitimdekiyle AYNI kuralla kırpar
(21 el noktasının kutusu → uzun kenar × 1.4 kare → 224×224, taşan kısım siyah).
Sunucu: kırpılmış eli models/ içindeki her eğitilmiş modele (int8 ONNX, sunucu CPU'sunda) aynı anda, paralel verir.
Sayfadaki "Tarayıcı" seçeneğinde modeller sunucuya hiç gitmeden tarayıcıda (ONNX Runtime Web) çalışır;
sunucu o zaman sadece .onnx dosyalarını verir.
Yeni bir model eğitildiğinde sayfayı yenilemek yeterli; sunucuyu yeniden başlatmaya gerek yok.

Kare kaydetme: sayfada 1 / 2 / 3 (Taş / Kağıt / Makas) tuşları ya da butonlarla o anki el, doğru etiketiyle
dataset_live/<sınıf>/ içine kaydedilir. Kaydedilen kare geniş "bağlam" karesidir (el kutusu × 2.4, 384 px,
tools/crop_context.py ile aynı); ortasındaki parça modelin gördüğü kırpmanın aynısıdır.
"""

import argparse
import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
PAGE = ROOT / "play.html"
HAND_MODEL = MODELS / "hand_landmarker.task"
ORDER = ("model_a", "model_b", "model_a_v2", "model_b_v2")
LIVE = ROOT / "dataset_live"
LABELS = ("rock", "paper", "scissors")
MAX_UPLOAD = 2 * 1024 * 1024

THREADS = 16  # model başına çekirdek (48 çekirdek var, modeller paralel çalışır)

_sessions = {}  # isim → (dosya değişme zamanı, oturum, meta, onnx yolu)
_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=len(ORDER))


def load_models():
    """models/<isim>.json + <isim>.onnx (int8; yoksa <isim>_fp32.onnx) olan modelleri yükler; değişenleri yeniden yükler.

    int8 sürüm fp32 ile aynı test F1'i alıyor ama Model B'de ~2 kat hızlı (CPU'da 28 → 16 ms, 8 çekirdekte).
    """
    found = []
    for name in ORDER:
        meta_path = MODELS / f"{name}.json"
        onnx_path = MODELS / f"{name}.onnx"
        if not onnx_path.exists():
            onnx_path = MODELS / f"{name}_fp32.onnx"
        if not (meta_path.exists() and onnx_path.exists()):
            continue
        mtime = onnx_path.stat().st_mtime
        with _lock:
            if name not in _sessions or _sessions[name][0] != mtime:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = THREADS
                sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
                _sessions[name] = (mtime, sess, json.loads(meta_path.read_text()), onnx_path)
        found.append(name)
    return found


def run_one(name, img):
    _, sess, meta, _ = _sessions[name]
    size = meta["input_size"]
    x = np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    x = (x - np.array(meta["mean"], dtype=np.float32)) / np.array(meta["std"], dtype=np.float32)
    x = x.transpose(2, 0, 1)[None]
    t = time.perf_counter()
    probs = sess.run(None, {"input": x})[0][0]
    return {"probs": [float(p) for p in probs], "ms": round((time.perf_counter() - t) * 1000, 1)}


def predict(jpeg):
    """Modeller paralel çalışır: cevap en yavaş model kadar sürer, toplamı kadar değil."""
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    names = load_models()
    jobs = {name: _pool.submit(run_one, name, img) for name in names}
    return {name: job.result() for name, job in jobs.items()}


def save(jpeg, label):
    """Etiketli bağlam karesini dataset_live/<sınıf>/ içine yazar; sınıf başına toplamı döndürür."""
    if label not in LABELS:
        raise ValueError("geçersiz etiket")
    Image.open(io.BytesIO(jpeg)).verify()  # gerçekten bir görüntü mü?
    dst = LIVE / label / f"{label}_live_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 10**6:06d}.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(jpeg)
    return {c: len(list((LIVE / c).glob("*.jpg"))) if (LIVE / c).exists() else 0 for c in LABELS}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type="application/json", cache="no-store"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _route(self):
        # Proxy ön ekini (/lab/proxy/8001/...) yok say, sadece son parçaya bak
        last = urlparse(self.path).path.rstrip("/").rsplit("/", 1)[-1]
        if last in ("models", "predict", "save", "hand_landmarker.task") or last.endswith(".onnx"):
            return last
        return ""

    def do_GET(self):
        route = self._route()
        if route == "":
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif route == "hand_landmarker.task":
            self._send(200, HAND_MODEL.read_bytes(), "application/octet-stream")
        elif route.endswith(".onnx") and route[:-5] in _sessions:
            # Tarayıcıda çalıştırma: modelin sunucuda kullanılan dosyası; tarayıcı önbelleğe alır (bir kez iner)
            _, _, _, path = _sessions[route[:-5]]
            self._send(200, path.read_bytes(), "application/octet-stream", "max-age=86400")  # URL ?v=<değişme zamanı> taşır
        elif route == "models":
            names = load_models()
            self._send(200, [{"name": n, **{k: _sessions[n][2][k] for k in ("model", "classes", "names_tr", "test_macro_f1", "crop_pad", "input_size", "mean", "std")},
                              "file": f"{n}.onnx?v={int(_sessions[n][0])}"}
                             for n in names])
        else:
            self._send(404, {"error": "bulunamadı"})

    def do_POST(self):
        route = self._route()
        if route not in ("predict", "save"):
            self._send(404, {"error": "bulunamadı"})
            return
        length = int(self.headers.get("Content-Length", 0))
        if not 0 < length <= MAX_UPLOAD:
            self._send(400, {"error": "geçersiz boyut"})
            return
        try:
            body = self.rfile.read(length)
            if route == "predict":
                self._send(200, predict(body))
            else:
                self._send(200, save(body, parse_qs(urlparse(self.path).query).get("label", [""])[0]))
        except Exception as e:
            self._send(400, {"error": str(e)})

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    names = load_models()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    proxy = os.environ.get("VSCODE_PROXY_URI")
    url = proxy.replace("{{port}}", str(args.port)) if proxy else f"http://localhost:{args.port}"
    print(f"Deneme sayfası: {url}")
    print(f"Yüklü modeller: {', '.join(names) or 'henüz yok (notebook ONNX dosyasını yazınca sayfayı yenile)'}")
    print("Durdurmak için Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
