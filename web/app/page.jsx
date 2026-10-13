"use client";
// Taş Kağıt Makas AI — her şey tarayıcıda: MediaPipe eli bulur, eğitilen model (ONNX Runtime Web) hareketi tanır.
// Kırpma kuralı eğitimdekiyle AYNI (tools/crop_hands.py): 21 el noktasının kutusu → uzun kenar × crop_pad kare
// → 224×224, görüntünün dışına taşan kısım siyah.
import { useEffect, useRef, useState } from "react";

const TR = { rock: "Taş", paper: "Kağıt", scissors: "Makas" };
const EMOJI = { rock: "✊", paper: "✋", scissors: "✌️" };
const COLOR = { rock: "var(--rock)", paper: "var(--paper)", scissors: "var(--scissors)" };
const SHOW = ["rock", "paper", "scissors"];
const BASE = process.env.NEXT_PUBLIC_BASE_PATH || "";   // public/ dosyalarının adresi (Vercel'de boş)
const TAU = 120, UNSURE = 0.6;   // üstel ortalama süresi (ms); en yüksek olasılık bunun altındaysa "Emin değilim"

// Paketlenmeden, public/ altından yüklenir (scripts/copy-assets.mjs)
const load = (url) => import(/* webpackIgnore: true */ url);

function squareBox(lms, w, h, pad) {
  const xs = lms.map((p) => p.x * w), ys = lms.map((p) => p.y * h);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const side = Math.max(x1 - x0, y1 - y0) * pad;
  return { x: (x0 + x1) / 2 - side / 2, y: (y0 + y1) / 2 - side / 2, side };
}

export default function Home() {
  const viewRef = useRef(null);
  const [status, setStatus] = useState({ text: "Yükleniyor…", kind: "load" });
  const [meta, setMeta] = useState(null);
  const [pred, setPred] = useState(null);       // { probs, ms } ya da el yoksa null
  const [hand, setHand] = useState(false);

  useEffect(() => {
    let stop = false, raf = 0, stream = null;
    (async () => {
      let m, ort, session, landmarker;
      try {
        setStatus({ text: "Model yükleniyor…", kind: "load" });
        m = await (await fetch(`${BASE}/models/model.json`)).json();
        setMeta(m);
        ort = await load(`${BASE}/ort/ort.wasm.min.mjs`);
        ort.env.wasm.wasmPaths = `${BASE}/ort/`;
        // Çok çekirdek sadece COOP/COEP başlıklarıyla açılır (next.config.mjs); yoksa tek çekirdek
        ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
        session = await ort.InferenceSession.create(`${BASE}/models/${m.file}`, { executionProviders: ["wasm"] });

        setStatus({ text: "El tespit modeli yükleniyor…", kind: "load" });
        const { HandLandmarker, FilesetResolver } = await load(`${BASE}/mediapipe/vision_bundle.mjs`);
        const fileset = await FilesetResolver.forVisionTasks(`${BASE}/mediapipe/wasm`);
        const make = (delegate) => HandLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: `${BASE}/models/hand_landmarker.task`, delegate }, runningMode: "VIDEO", numHands: 1,
        });
        landmarker = await make("GPU").catch(() => make("CPU"));
      } catch (e) {
        setStatus({ text: "Model yüklenemedi. Sayfayı yenilemeyi dene.", kind: "err", detail: e.message });
        return;
      }

      setStatus({ text: "Kamera açılıyor…", kind: "load" });
      const video = document.createElement("video");
      video.playsInline = true; video.muted = true;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false,
        });
        video.srcObject = stream;
        await video.play();
      } catch {
        setStatus({ text: "Kameraya erişilemedi. Lütfen izin ver.", kind: "err" });
        return;
      }
      if (stop) return;
      setStatus({ text: "Canlı", kind: "ok" });

      const view = viewRef.current, ctx = view.getContext("2d");
      const crop = document.createElement("canvas");
      crop.width = crop.height = m.input_size;
      const cctx = crop.getContext("2d", { willReadFrequently: true });
      const S = m.input_size, x = new Float32Array(3 * S * S);
      let probs = null, lastT = 0, busy = false, ms = 0, box = null, lastUi = 0;

      const run = async () => {
        busy = true;
        cctx.fillStyle = "#000"; cctx.fillRect(0, 0, S, S);
        cctx.drawImage(video, box.x, box.y, box.side, box.side, 0, 0, S, S);   // taşan kısım siyah kalır
        const d = cctx.getImageData(0, 0, S, S).data;
        for (let i = 0; i < S * S; i++)
          for (let c = 0; c < 3; c++) x[c * S * S + i] = (d[i * 4 + c] / 255 - m.mean[c]) / m.std[c];
        const t = performance.now();
        try {
          const out = await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", x, [1, 3, S, S]) });
          const n = Array.from(out[session.outputNames[0]].data), now = performance.now();
          ms = ms ? 0.8 * ms + 0.2 * (now - t) : now - t;
          if (box) {
            const k = lastT ? 1 - Math.exp(-(now - lastT) / TAU) : 1;
            probs = probs ? probs.map((v, i) => v + k * (n[i] - v)) : n;
            lastT = now;
          }
        } catch (e) {
          setStatus({ text: "Tahmin yapılamadı.", kind: "err", detail: e.message });
        }
        busy = false;
      };

      const frame = () => {
        if (stop) return;
        const W = video.videoWidth, H = video.videoHeight;
        if (W && video.readyState >= 2) {
          if (view.width !== W) { view.width = W; view.height = H; }
          const res = landmarker.detectForVideo(video, performance.now());
          box = res.landmarks.length ? squareBox(res.landmarks[0], W, H, m.crop_pad) : null;
          if (!box) { probs = null; lastT = 0; }
          else if (!busy) run();

          // ayna görüntü + kutu
          ctx.save(); ctx.translate(W, 0); ctx.scale(-1, 1); ctx.drawImage(video, 0, 0, W, H); ctx.restore();
          if (box && probs) {
            const b = probs.indexOf(Math.max(...probs)), cls = m.classes[b], sure = probs[b] >= UNSURE;
            const color = getComputedStyle(document.documentElement).getPropertyValue(sure ? `--${cls}` : "--unsure").trim();
            const bx = W - box.x - box.side, lw = Math.max(3, W / 150), fs = Math.round(W / 22);
            ctx.lineWidth = lw; ctx.strokeStyle = color;
            ctx.beginPath(); ctx.roundRect(bx, box.y, box.side, box.side, box.side * 0.08); ctx.stroke();
            const label = sure ? `${TR[cls]} %${Math.round(probs[b] * 100)}` : "Emin değilim";
            ctx.font = `800 ${fs}px system-ui`;
            const tw = ctx.measureText(label).width + fs, th = fs * 1.6, ty = Math.max(4, box.y - th - 8);
            ctx.fillStyle = color; ctx.beginPath(); ctx.roundRect(bx, ty, tw, th, th / 2); ctx.fill();
            ctx.fillStyle = "#fff"; ctx.fillText(label, bx + fs / 2, ty + th / 2 + fs * 0.36);
          }
          const now = performance.now();
          if (now - lastUi > 80) {   // React'i her karede değil, saniyede ~12 kez güncelle
            lastUi = now;
            setHand(!!box);
            setPred(box && probs ? { probs: [...probs], ms } : null);
          }
        }
        raf = requestAnimationFrame(frame);
      };
      frame();
    })();
    return () => { stop = true; cancelAnimationFrame(raf); stream?.getTracks().forEach((t) => t.stop()); };
  }, []);

  let cls = null, top = 0, sure = false;
  if (pred && meta) {
    const b = pred.probs.indexOf(Math.max(...pred.probs));
    cls = meta.classes[b]; top = pred.probs[b]; sure = top >= UNSURE;
  }
  const live = status.kind === "ok";

  return (
    <main>
      <header>
        <div className="logo raised" aria-hidden="true"><span /></div>
        <div>
          <h1>Taş Kağıt Makas AI</h1>
          <p>Elini göster, ne yaptığını tahmin edeyim.</p>
        </div>
        <div className={`chip raised ${live ? "ok" : status.kind === "err" ? "err" : ""}`} role="status">
          <span className="dot" />{status.text}
        </div>
      </header>

      <div className="layout">
        <section className="stage raised">
          <div className="view">
            <canvas ref={viewRef} width={640} height={480} aria-label="Kamera görüntüsü" />
            {!live && (
              <div className={`overlay ${status.kind === "err" ? "err" : ""}`}>
                <div>
                  {status.kind === "load" && <div className="spinner" />}
                  {status.text}
                  {status.detail && <div style={{ fontSize: 13, fontWeight: 500, opacity: .8, marginTop: 8 }}>{status.detail}</div>}
                </div>
              </div>
            )}
            {live && !hand && <div className="overlay">✋ Elini kameraya göster</div>}
          </div>
        </section>

        <aside className="side">
          <section className="card raised" aria-live="polite">
            <h2>Tahmin</h2>
            <div className="big" style={{ color: sure ? COLOR[cls] : "var(--unsure)" }}>
              {sure ? `${EMOJI[cls]} ${TR[cls]}` : pred ? "🤔 Emin değilim" : "—"}
            </div>
            <div className="conf">{pred ? `Güven %${Math.round(top * 100)}` : live ? "Elini kameraya göster" : "Hazırlanıyor…"}</div>
            <div className="bars">
              {SHOW.map((c) => {
                const v = pred && meta ? pred.probs[meta.classes.indexOf(c)] : 0;
                return (
                  <div className="bar" key={c}>
                    <span>{TR[c]}</span>
                    <div className="track inset"><i style={{ width: `${v * 100}%`, background: COLOR[c] }} /></div>
                    <span>{pred ? `%${Math.round(v * 100)}` : "–"}</span>
                  </div>
                );
              })}
            </div>
          </section>

          <section className="card raised">
            <h2>Model</h2>
            <dl className="info">
              <dt>Ad</dt><dd>{meta?.model ?? "–"}</dd>
              <dt>Test macro F1</dt><dd>{meta ? meta.test_macro_f1.toFixed(3) : "–"}</dd>
              <dt>Tahmin süresi</dt><dd>{pred ? `${Math.round(pred.ms)} ms` : "–"}</dd>
              <dt>Çalıştığı yer</dt><dd>Tarayıcın</dd>
            </dl>
          </section>

          <div className="tip raised">✋ <span>Elinin tamamı kadrajda olsun; hareketi net yap.</span></div>
        </aside>
      </div>
    </main>
  );
}
