# %% [markdown]
# # DigitAgent – Gemma-3 27B-IT Colab Notebook
#
# Google Generative AI API ile **Gemma-3 27B-IT (multimodal)** modelini
# kullanarak Drive’daki PNG **digit** görsellerini çözer;
# dosyaları `<digits>.png` adıyla `renamed/` klasörüne kopyalar.
# ---------------------------------------------------------------
# ⚠️  API anahtarınızı koda yazmayın!  Runtime ▸ “Change env.” ile
#     GOOGLE_API_KEY değişkeni olarak ekleyin.

# %% 🔧 Kurulum
!pip -q install --upgrade google-generativeai pillow tqdm

# %% 🔑 API anahtarı ve model
import os, google.generativeai as genai

API_KEY = os.getenv("GOOGLE_API_KEY")  # Colab’de Environment variable
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY ortam değişkeni tanımlanmalı!")

genai.configure(api_key=API_KEY)
MODEL_NAME = "models/gemma-3-27b-it"   # Gemma-3 Vision 27B-IT
model = genai.GenerativeModel(MODEL_NAME)
print("🟢 Model yüklendi →", MODEL_NAME)

# %% 🔗 Google Drive bağlantısı
from google.colab import drive
drive.mount("/content/drive")

# %% 🧩 Yardımcılar
from pathlib import Path
from PIL import Image
import re, json, shutil
from tqdm.auto import tqdm

PROMPT = (
    "<start_of_image>\nYou are a helpful vision agent.\n"
    "Task: What are the numbers?\n"
    'Respond ONLY in JSON like {"digits": "123456"}'
)
_rx = re.compile(r'"digits"\s*:\s*"(\d+)"')

def parse_digits(txt: str):
    try:
        return str(json.loads(txt).get("digits"))
    except Exception:
        m = _rx.search(txt); return m.group(1) if m else None

def solve_digit(img_path: Path):
    reply = model.generate_content(
        [PROMPT, Image.open(img_path)],
        generation_config={"max_output_tokens": 20, "temperature": 0.1},
        stream=False,
    ).text
    return parse_digits(reply)

def copy_unique(src: Path, dst_dir: Path, digits: str):
    dst_dir.mkdir(parents=True, exist_ok=True)
    i = 0
    while True:
        name = f"{digits}{f'_{i}' if i else ''}{src.suffix}"
        dst = dst_dir / name
        if not dst.exists():
            shutil.copy2(src, dst); return dst
        i += 1

# %% 🚀 Klasördeki tüm digit görsellerini çöz
SRC_DIR = Path("/content/drive/MyDrive/digitsolver/digits")   # kaynak
DST_DIR = Path("/content/drive/MyDrive/digitsolver/renamed")  # hedef

done = skipped = 0
for img in tqdm(sorted(SRC_DIR.glob("*.png")), desc="DIGIT"):
    digits = solve_digit(img)
    if digits:
        copy_unique(img, DST_DIR, digits); done += 1
    else:
        skipped += 1
print(f"✅ {done} dosya çözüldü, ⚠️ {skipped} atlandı.")

# %% 📁 Örnek çıktı listesi
for f in list(DST_DIR.glob("*.png"))[:10]:
    print(f.name)
print("Toplam:", len(list(DST_DIR.glob('*.png'))))
