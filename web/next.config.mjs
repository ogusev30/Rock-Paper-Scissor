// COOP + COEP: sayfa "cross-origin isolated" olur → ONNX Runtime Web birden çok çekirdekle (WASM thread) çalışır.
// Bütün dosyalar (model, ort, mediapipe) aynı adresten geldiği için bu başlıklar hiçbir şeyi engellemez.
// BASE_PATH: sadece sunucudaki önizleme için (code-server: /lab/absproxy/<port>); Vercel'de boş kalır.
const basePath = process.env.BASE_PATH || "";
const nextConfig = {
  basePath,
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
  async headers() {
    return [{
      source: "/:path*",
      headers: [
        { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
        { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
      ],
    }];
  },
};
export default nextConfig;
