"""İyileştirme deneyleri: yeni foto olmadan Model A ve Model B'yi ilerletmek.

Kullanım:
    .venv/bin/python improve.py run A_new --seed 1 --threads 16   # bir deney
    .venv/bin/python improve.py baseline                         # training.ipynb modellerini zor testte ölç
    .venv/bin/python improve.py summary                          # tüm sonuçlar tablo olarak
    .venv/bin/python improve.py export A_new --seed 1 --name model_a_v2

Neden?
    Asıl test seti (73 foto) doldu: A 0.987, B 1.000; aradaki fark tek fotoğraf. Bu yüzden ölçüm için
    "zor test" eklenir: aynı val / test fotoğraflarının kameradaki gibi bozulmuş halleri (kutu kayması,
    farklı kırpma payı, karanlık, gürültü, hareket bulanıklığı, JPEG, düşük çözünürlük, eğik el).
    Model ve tarif seçimi sadece val ile yapılır; test sadece raporlanır.

Tarifler (iki modelde aynı):
    base  training.ipynb'deki augmentation (kırpılmış 224 px el üstünde)
    new   bağlam karesinden (tools/crop_context.py) rastgele pay 1.2–1.7 ve kaymayla kesme + döndürme,
          renk, ve kamera bozulmaları (hareket bulanıklığı, JPEG, gürültü, düşük çözünürlük, karanlık) + RandomErasing
"""

import argparse
import copy
import io
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import f1_score
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
SPLIT, CONTEXT = ROOT / "dataset_split", ROOT / "dataset_context"
MODELS, OUT = ROOT / "models", ROOT / "results" / "v2"
CKPT = MODELS / "v2"
IMG, CTX_PAD, PAD = 224, 2.4, 1.4
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CLASSES = ["paper", "rock", "scissors"]
TR = {"rock": "Taş", "paper": "Kağıt", "scissors": "Makas"}
normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


# ---------------------------------------------------------------- görüntü işlemleri

def cut(ctx, pad=PAD, dx=0.0, dy=0.0, angle=0.0):
    """Bağlam karesinden (384 px, el kutusu × 2.4) el kırpması: pay, kayma (kırpma kenarının oranı), açı."""
    S = ctx.width
    if angle:
        ctx = ctx.rotate(angle, resample=Image.BILINEAR)  # taşan köşeler siyah
    side = S * pad / CTX_PAD
    cx, cy = S / 2 + dx * side, S / 2 + dy * side
    return ctx.crop((cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)).resize((IMG, IMG), Image.BILINEAR)


def motion_blur(im, k, angle):
    kern = np.zeros((k, k), np.float32)
    kern[k // 2] = 1.0
    kern = cv2.warpAffine(kern, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0), (k, k))
    return Image.fromarray(cv2.filter2D(np.asarray(im), -1, kern / kern.sum()))


def jpeg(im, q):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def noise(im, sigma, rng):
    a = np.asarray(im, np.float32) / 255 + rng.normal(0, sigma, (im.height, im.width, 3))
    return Image.fromarray((a.clip(0, 1) * 255).astype(np.uint8))


def lowres(im, px):
    return im.resize((px, px), Image.BILINEAR).resize((IMG, IMG), Image.BILINEAR)


def dark(im, f):
    a = (np.asarray(im, np.float32) / 255) ** 1.3 * f  # loş ışık: koyu tonlar daha da kararır
    return Image.fromarray((a.clip(0, 1) * 255).astype(np.uint8))


# ---------------------------------------------------------------- datasetler

def load_context(split):
    items = []
    for i, c in enumerate(CLASSES):
        for p in sorted((CONTEXT / split / c).glob("*.jpg")):
            items.append((Image.open(p).convert("RGB"), i))
    return items


class NewTrain(torch.utils.data.Dataset):
    """Yeni tarif: her seferinde bağlam karesinden farklı bir kırpma + kamera bozulmaları."""
    jitter = transforms.ColorJitter(brightness=0.35, contrast=0.3, saturation=0.3, hue=0.03)
    erase = transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), value=0)

    def __init__(self, strong=False):
        self.strong = strong  # new2: daha geniş pay / kayma / parlaklık + %10 siyah-beyaz
        self.items = load_context("train")
        self.targets = [y for _, y in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ctx, y = self.items[i]
        r = random.random
        pad, sh = ((1.15, 1.8), 0.14) if self.strong else ((1.2, 1.7), 0.1)
        im = cut(ctx, pad=random.uniform(*pad), dx=random.uniform(-sh, sh),
                 dy=random.uniform(-sh, sh), angle=random.uniform(-25, 25))
        if r() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im = self.jitter(im)
        if self.strong and r() < 0.2:
            im = ImageEnhance.Brightness(im).enhance(random.uniform(1.3, 1.7))
        if self.strong and r() < 0.1:
            im = im.convert("L").convert("RGB")
        if r() < 0.25:
            im = dark(im, random.uniform(0.45, 0.9))
        if r() < 0.2:
            im = motion_blur(im, random.choice([3, 5, 7]), random.uniform(0, 180))
        elif r() < 0.15:
            im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.5)))
        if r() < 0.2:
            im = lowres(im, random.randint(64, 128))
        if r() < 0.2:
            im = noise(im, random.uniform(0.01, 0.05), np.random)
        if r() < 0.3:
            im = jpeg(im, random.randint(30, 80))
        return self.erase(normalize(im)), y


def base_train():
    """training.ipynb'deki augmentation, kırpılmış el üstünde."""
    tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.35, contrast=0.3, saturation=0.3, hue=0.03),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    return datasets.ImageFolder(SPLIT / "train", transform=tf)


# Zor test: her biri eğitimdeki aralığın kenarında ya da biraz dışında
STRESS = {
    "kutu küçük (pay 1.1)":     lambda c, g: cut(c, pad=1.1),
    "kutu büyük (pay 1.9)":     lambda c, g: cut(c, pad=1.9),
    "kayma sağ-aşağı":          lambda c, g: cut(c, dx=0.15, dy=0.15),
    "kayma sol-yukarı":         lambda c, g: cut(c, dx=-0.15, dy=-0.15),
    "eğik el (35°)":            lambda c, g: cut(c, angle=35),
    "karanlık":                 lambda c, g: dark(cut(c), 0.35),
    "parlak":                   lambda c, g: ImageEnhance.Brightness(cut(c)).enhance(1.8),
    "gürültü":                  lambda c, g: noise(cut(c), 0.08, g),
    "hareket bulanıklığı":      lambda c, g: motion_blur(cut(c), 11, 30),
    "JPEG kalite 15":           lambda c, g: jpeg(cut(c), 15),
    "düşük çözünürlük (48 px)": lambda c, g: lowres(cut(c), 48),
    "siyah-beyaz":              lambda c, g: cut(c).convert("L").convert("RGB"),
}


def stress_set(split):
    """{bozulma adı: (x, y)} — sabit (rastgelelik seed'li), her çalıştırmada aynı."""
    items, out = load_context(split), {}
    y = torch.tensor([t for _, t in items])
    for name, fn in STRESS.items():
        g = np.random.default_rng(0)
        out[name] = (torch.stack([normalize(fn(c, g)) for c, _ in items]), y)
    return out


def clean_set(split):
    ds = datasets.ImageFolder(SPLIT / split, transform=normalize)
    assert ds.classes == CLASSES
    return torch.stack([x for x, _ in ds]), torch.tensor(ds.targets)


# ---------------------------------------------------------------- modeller

def model_a(head="linear"):
    """Onaylanan Model A (training.ipynb). head="gap": büyük Linear yerine Global Average Pooling."""
    def block(c_in, c_out):
        return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2))
    convs = [block(3, 32), block(32, 64), block(64, 128), block(128, 128), block(128, 256)]
    if head == "gap":
        return nn.Sequential(*convs, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, len(CLASSES)))
    return nn.Sequential(*convs, nn.Flatten(), nn.Dropout(0.5), nn.Linear(256 * 7 * 7, 2000), nn.ReLU(),
                         nn.Dropout(0.5), nn.Linear(2000, len(CLASSES)))


def model_b():
    import timm
    return timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=True, num_classes=len(CLASSES))


def set_trainable(model, parts):
    for p in model.parameters():
        p.requires_grad = False
    for part in parts:
        for p in part.parameters():
            p.requires_grad = True


# Her deney: model, tarif ve eğitim adımları (parçalar, epoch, lr, weight decay)
CONFIGS = {
    "A_base": dict(model=lambda: model_a(), data="base", steps=lambda m: [(None, 60, 1e-3, 1e-4)]),
    "A_new":  dict(model=lambda: model_a(), data="new", steps=lambda m: [(None, 60, 1e-3, 1e-4)]),
    "A_gap":  dict(model=lambda: model_a("gap"), data="new", steps=lambda m: [(None, 100, 2e-3, 1e-4)]),
    # Yeni tarif daha zor: 60 / 100 epoch sonunda loss hâlâ düşüyordu → daha uzun eğitim
    "A_new_long": dict(model=lambda: model_a(), data="new", steps=lambda m: [(None, 120, 1e-3, 1e-4)]),
    "A_gap_long": dict(model=lambda: model_a("gap"), data="new", steps=lambda m: [(None, 160, 2e-3, 1e-4)]),
    # new2: zor testte zayıf kalan siyah-beyaz / parlak / kayma için geniş tarif; A_new_long 112/120'de hâlâ iniyordu
    "A_new2_long": dict(model=lambda: model_a(), data="new2", steps=lambda m: [(None, 200, 1e-3, 1e-4)]),
    "B_base": dict(model=model_b, data="base",
                   steps=lambda m: [([m.head], 5, 1e-3, 0.05), ([m.stages[-1], m.head], 10, 1e-4, 0.05)]),
    "B_new":  dict(model=model_b, data="new",
                   steps=lambda m: [([m.head], 5, 1e-3, 0.05), ([m.stages[-1], m.head], 10, 1e-4, 0.05)]),
    "B_deep": dict(model=model_b, data="new",
                   steps=lambda m: [([m.head], 3, 1e-3, 0.05), ([m.stages[2], m.stages[3], m.head], 15, 5e-5, 0.05)]),
    "B_deep2": dict(model=model_b, data="new2",
                    steps=lambda m: [([m.head], 3, 1e-3, 0.05), ([m.stages[2], m.stages[3], m.head], 15, 5e-5, 0.05)]),
}


# ---------------------------------------------------------------- eğitim ve ölçüm

@torch.no_grad()
def probs(model, x, bs=128):
    model.eval()
    return torch.cat([model(x[i:i + bs]).softmax(1) for i in range(0, len(x), bs)])


def scores(p, y):
    return dict(f1=float(f1_score(y, p.argmax(1), average="macro")),
                nll=float(F.nll_loss(p.clamp_min(1e-7).log(), y)))


def report(model, sets):
    """{set adı: {f1, nll}} + zor testin ortalaması."""
    out = {name: scores(probs(model, x), y) for name, (x, y) in sets.items()}
    hard = [v for k, v in out.items() if k != "temiz"]
    out["zor ortalama"] = dict(f1=float(np.mean([v["f1"] for v in hard])), nll=float(np.mean([v["nll"] for v in hard])))
    return out


def val_loss(model, val_sets):
    """Model seçimi: temiz + zor val kümelerinin ortalama loss'u (sadece val, test asla)."""
    return float(np.mean([F.nll_loss(probs(model, x).clamp_min(1e-7).log(), y).item() for x, y in val_sets.values()]))


def run(name, seed, threads):
    torch.set_num_threads(threads)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    cfg = CONFIGS[name]
    train_ds = NewTrain(cfg["data"] == "new2") if cfg["data"] != "base" else base_train()
    dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0,
                                     generator=torch.Generator().manual_seed(seed))
    val_sets = {"temiz": clean_set("val"), **stress_set("val")}
    model = cfg["model"]()
    best, best_state, hist, t0 = math.inf, None, [], time.time()
    for parts, epochs, lr, wd in cfg["steps"](model):
        if parts is None:
            set_trainable(model, [model])
        else:
            set_trainable(model, parts)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(dl))
        for _ in range(epochs):
            model.train()
            total = 0.0
            for x, y in dl:
                loss = F.cross_entropy(model(x), y, label_smoothing=0.1)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                total += loss.item() * len(y)
            vl = val_loss(model, val_sets)
            hist.append(dict(epoch=len(hist) + 1, train_loss=total / len(train_ds), val_loss=vl))
            if vl < best:
                best, best_state = vl, copy.deepcopy(model.state_dict())
            print(f"{name} s{seed} epoch {len(hist):>3} | train loss {hist[-1]['train_loss']:.3f} | val (temiz+zor) loss {vl:.3f}",
                  flush=True)
    model.load_state_dict(best_state)
    CKPT.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT / f"{name}_s{seed}.pt")
    res = dict(config=name, seed=seed, minutes=(time.time() - t0) / 60, best_val_loss=best,
               val=report(model, val_sets), test=report(model, {"temiz": clean_set("test"), **stress_set("test")}),
               history=hist)
    (OUT / f"{name}_s{seed}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"BITTI {name} s{seed}: test temiz F1 {res['test']['temiz']['f1']:.3f} | test zor F1 {res['test']['zor ortalama']['f1']:.3f}"
          f" | val zor F1 {res['val']['zor ortalama']['f1']:.3f} | {res['minutes']:.1f} dk", flush=True)


def baseline():
    """training.ipynb'de eğitilmiş model_a / model_b (seçim: val F1) aynı zor testte."""
    import timm  # noqa: F401
    torch.set_num_threads(48)
    OUT.mkdir(parents=True, exist_ok=True)
    val_sets = {"temiz": clean_set("val"), **stress_set("val")}
    test_sets = {"temiz": clean_set("test"), **stress_set("test")}
    for name, make in (("model_a", model_a), ("model_b", model_b)):
        m = make()
        m.load_state_dict(torch.load(MODELS / f"{name}_best.pt"))
        res = dict(config=f"{name} (training.ipynb)", seed=42, val=report(m, val_sets), test=report(m, test_sets))
        (OUT / f"{name}_notebook.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(f"{name}: test temiz F1 {res['test']['temiz']['f1']:.3f} | test zor F1 {res['test']['zor ortalama']['f1']:.3f}"
              f" | val zor F1 {res['val']['zor ortalama']['f1']:.3f}")


def summary():
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    groups = {}
    for r in rows:
        groups.setdefault(r["config"], []).append(r)
    ms = lambda v: f"{np.mean(v):.3f} ± {np.std(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"
    print(f"{'deney':<26}{'n':>3}  {'val zor F1':>15}  {'val zor loss':>15}  {'test temiz F1':>15}  {'test zor F1':>15}  {'test zor loss':>15}")
    for name, g in groups.items():
        get = lambda s, k, m: [r[s][k][m] for r in g]
        print(f"{name:<26}{len(g):>3}  {ms(get('val', 'zor ortalama', 'f1')):>15}  {ms(get('val', 'zor ortalama', 'nll')):>15}"
              f"  {ms(get('test', 'temiz', 'f1')):>15}  {ms(get('test', 'zor ortalama', 'f1')):>15}  {ms(get('test', 'zor ortalama', 'nll')):>15}")
    print("\nBozulma başına test F1 (seed ortalaması):")
    names = list(STRESS)
    print(f"{'':<26}" + "".join(f"{n[:11]:>12}" for n in names))
    for name, g in groups.items():
        print(f"{name:<26}" + "".join(f"{np.mean([r['test'][n]['f1'] for r in g]):>12.3f}" for n in names))


class WithSoftmax(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        return self.m(x).softmax(1)


def export(name, seed, out_name, title):
    """training.ipynb'deki export_onnx ile aynı: <out>_fp32.onnx + int8 <out>.onnx + <out>.json."""
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    model = CONFIGS[name]["model"]()
    model.load_state_dict(torch.load(CKPT / f"{name}_s{seed}.pt"))
    wrapped = WithSoftmax(model).eval()
    fp32, small = MODELS / f"{out_name}_fp32.onnx", MODELS / f"{out_name}.onnx"
    torch.onnx.export(wrapped, torch.randn(1, 3, IMG, IMG), fp32, input_names=["input"], output_names=["probs"],
                      dynamic_axes={"input": {0: "batch"}, "probs": {0: "batch"}}, opset_version=17, dynamo=False)
    wrapped.eval()
    quantize_dynamic(str(fp32), str(small), op_types_to_quantize=["MatMul", "Gemm"], weight_type=QuantType.QInt8)
    x, y = clean_set("test")
    ref = probs(wrapped.m, x).numpy()
    print(f"{'dosya':<24}{'boyut':>9}{'PyTorch ile fark':>18}{'test macro F1':>15}")
    for path in (fp32, small):
        got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"input": x.numpy()})[0]
        print(f"{path.name:<24}{path.stat().st_size / 1e6:>7.1f}MB{np.abs(ref - got).max():>18.2e}"
              f"{f1_score(y, got.argmax(1), average='macro'):>15.3f}")
    res = json.loads((OUT / f"{name}_s{seed}.json").read_text())
    meta = dict(model=title, file=small.name, classes=CLASSES, names_tr=[TR[c] for c in CLASSES],
                input_size=IMG, mean=MEAN, std=STD, crop_pad=PAD, test_macro_f1=round(res["test"]["temiz"]["f1"], 4),
                test_stress_f1=round(res["test"]["zor ortalama"]["f1"], 4), recipe=f"improve.py {name} seed {seed}")
    (MODELS / f"{out_name}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "baseline", "summary", "export"])
    ap.add_argument("config", nargs="?", choices=list(CONFIGS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--name"), ap.add_argument("--title")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.config, a.seed, a.threads)
    elif a.cmd == "baseline":
        baseline()
    elif a.cmd == "summary":
        summary()
    else:
        export(a.config, a.seed, a.name, a.title or a.config)
