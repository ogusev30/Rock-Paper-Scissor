// ONNX Runtime Web ve MediaPipe'ın çalışma dosyalarını public/ altına kopyalar (npm install'dan sonra, her build'de).
// Böylece her şey aynı adresten gelir: CDN'e bağımlılık yok ve COOP/COEP ile çok çekirdekli WASM açılabilir.
import { cpSync, mkdirSync } from "node:fs";

const copy = (from, to, files) => {
  mkdirSync(to, { recursive: true });
  for (const f of files) cpSync(`${from}/${f}`, `${to}/${f}`);
};
copy("node_modules/onnxruntime-web/dist", "public/ort",
     ["ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]);
copy("node_modules/@mediapipe/tasks-vision", "public/mediapipe", ["vision_bundle.mjs"]);
copy("node_modules/@mediapipe/tasks-vision/wasm", "public/mediapipe/wasm",
     ["vision_wasm_internal.js", "vision_wasm_internal.wasm", "vision_wasm_nosimd_internal.js", "vision_wasm_nosimd_internal.wasm"]);
console.log("ort + mediapipe dosyaları public/ altına kopyalandı");
