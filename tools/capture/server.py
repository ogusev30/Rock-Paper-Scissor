"""Taş-Kağıt-Makas çekim sayfası sunucusu.

Kullanım:
    python tools/capture/server.py            # terminalde yazan linki aç
    python tools/capture/server.py --port 8080

Laptop kamerasından çekilen fotoğrafları dataset/<sınıf>/ klasörüne kaydeder.
Sadece Python standart kütüphanesini kullanır.
"""

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"
INDEX = Path(__file__).with_name("index.html")
LABELS = ("rock", "paper", "scissors")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
MAX_UPLOAD = 10 * 1024 * 1024


def images(label):
    folder = DATASET / label
    return sorted(
        (p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: p.stat().st_mtime,
    )


def counts():
    return {label: len(images(label)) for label in LABELS}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _route(self):
        # Proxy ön ekini (/lab/proxy/8000/...) yok say, sadece son parçaya bak
        last = urlparse(self.path).path.rstrip("/").rsplit("/", 1)[-1]
        return "/" + last if last in ("counts", "save", "undo") else "/"

    def _label(self):
        label = parse_qs(urlparse(self.path).query).get("label", [""])[0]
        return label if label in LABELS else None

    def do_GET(self):
        path = self._route()
        if path == "/":
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        elif path == "/counts":
            self._send(200, counts())
        else:
            self._send(404, {"error": "bulunamadı"})

    def do_POST(self):
        path = self._route()
        label = self._label()
        if label is None:
            self._send(400, {"error": f"label şunlardan biri olmalı: {', '.join(LABELS)}"})
            return

        if path == "/save":
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= MAX_UPLOAD:
                self._send(400, {"error": "geçersiz dosya boyutu"})
                return
            data = self.rfile.read(length)
            if not data.startswith(b"\xff\xd8"):
                self._send(400, {"error": "JPEG bekleniyordu"})
                return
            name = f"{label}_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000:06d}.jpg"
            (DATASET / label / name).write_bytes(data)
            self._send(200, {"saved": name, "counts": counts()})

        elif path == "/undo":
            files = [p for p in images(label) if p.name.startswith(f"{label}_")]
            if not files:
                self._send(404, {"error": "silinecek fotoğraf yok", "counts": counts()})
                return
            files[-1].unlink()
            self._send(200, {"deleted": files[-1].name, "counts": counts()})

        else:
            self._send(404, {"error": "bulunamadı"})

    def log_message(self, fmt, *args):
        if "/counts" not in self.path:
            super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    for label in LABELS:
        (DATASET / label).mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    proxy = os.environ.get("VSCODE_PROXY_URI")
    url = proxy.replace("{{port}}", str(args.port)) if proxy else f"http://localhost:{args.port}"
    print(f"Çekim sayfası: {url}")
    print(f"Fotoğraflar buraya kaydediliyor: {DATASET}")
    print("Durdurmak için Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
