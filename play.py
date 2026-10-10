"""Aşama 5: Deneme sayfası — eğitilen modelleri kendi elinle kamerada dene.

Kullanım:
    .venv/bin/python play.py            # terminalde yazan linki aç

Tarayıcı: kamerayı açar, MediaPipe ile eli bulur ve eğitimdekiyle AYNI kuralla kırpar
(21 el noktasının kutusu → uzun kenar × 1.4 kare → 224×224, taşan kısım siyah).
Sunucu: kırpılmış eli models/ içindeki her eğitilmiş modele (fp32 ONNX, sunucu CPU'sunda) verir.
Yeni bir model eğitildiğinde sayfayı yenilemek yeterli; sunucuyu yeniden başlatmaya gerek yok.
"""

import argparse
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
PAGE = ROOT / "play.html"
HAND_MODEL = MODELS / "hand_landmarker.task"
ORDER = ("model_a", "model_b")
MAX_UPLOAD = 2 * 1024 * 1024

_sessions = {}  # isim → (dosya değişme zamanı, oturum, meta)
_lock = threading.Lock()


def load_models():
    """models/<isim>.json + <isim>_fp32.onnx (yoksa <isim>.onnx) olan modelleri yükler; değişenleri yeniden yükler."""
    found = []
    for name in ORDER:
        meta_path = MODELS / f"{name}.json"
        onnx_path = MODELS / f"{name}_fp32.onnx"
        if not onnx_path.exists():
            onnx_path = MODELS / f"{name}.onnx"
        if not (meta_path.exists() and onnx_path.exists()):
            continue
        mtime = onnx_path.stat().st_mtime
        with _lock:
            if name not in _sessions or _sessions[name][0] != mtime:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 8
                sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
                _sessions[name] = (mtime, sess, json.loads(meta_path.read_text()))
        found.append(name)
    return found


def predict(jpeg):
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    out = {}
    for name in load_models():
        _, sess, meta = _sessions[name]
        size = meta["input_size"]
        x = np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
        x = (x - np.array(meta["mean"], dtype=np.float32)) / np.array(meta["std"], dtype=np.float32)
        x = x.transpose(2, 0, 1)[None]
        probs = sess.run(None, {"input": x})[0][0]
        out[name] = {"probs": [float(p) for p in probs]}
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _route(self):
        # Proxy ön ekini (/lab/proxy/8001/...) yok say, sadece son parçaya bak
        last = urlparse(self.path).path.rstrip("/").rsplit("/", 1)[-1]
        return last if last in ("models", "predict", "hand_landmarker.task") else ""

    def do_GET(self):
        route = self._route()
        if route == "":
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif route == "hand_landmarker.task":
            self._send(200, HAND_MODEL.read_bytes(), "application/octet-stream")
        elif route == "models":
            names = load_models()
            self._send(200, [{"name": n, **{k: _sessions[n][2][k] for k in ("model", "classes", "names_tr", "test_macro_f1", "crop_pad")}}
                             for n in names])
        else:
            self._send(404, {"error": "bulunamadı"})

    def do_POST(self):
        if self._route() != "predict":
            self._send(404, {"error": "bulunamadı"})
            return
        length = int(self.headers.get("Content-Length", 0))
        if not 0 < length <= MAX_UPLOAD:
            self._send(400, {"error": "geçersiz boyut"})
            return
        try:
            self._send(200, predict(self.rfile.read(length)))
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
