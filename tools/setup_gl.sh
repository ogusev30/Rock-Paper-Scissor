#!/usr/bin/env bash
# MediaPipe (1.x) Linux'ta libEGL.so.1 ister; sunucuda yok ve root yetkimiz yok.
# Debian'ın libglvnd paketlerini indirip vendor/gl/ içine açar (root gerekmez).
# Kullanım: bash tools/setup_gl.sh
set -euo pipefail
cd "$(dirname "$0")/.."
arch=$(dpkg --print-architecture)   # arm64 veya amd64
triplet=$(gcc -dumpmachine 2>/dev/null || echo "$(uname -m)-linux-gnu")
tmp=$(mktemp -d)
for p in libegl1 libglvnd0 libgles2; do
  curl -sSf -o "$tmp/$p.deb" "http://deb.debian.org/debian/pool/main/libg/libglvnd/${p}_1.7.0-3+b1_${arch}.deb"
  dpkg-deb -x "$tmp/$p.deb" "$tmp/root"
done
rm -rf vendor/gl && mkdir -p vendor
cp -a "$tmp/root/usr/lib/$(uname -m)-linux-gnu" vendor/gl
rm -rf "$tmp"
echo "Hazır: $(ls vendor/gl | tr '\n' ' ')"
