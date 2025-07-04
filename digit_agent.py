#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DigitAgent – number-CAPTCHA renaming agent
=========================================

Klasördeki PNG dosyalarının üzerindeki rakamları
Gemma-3 4B IT modeliyle çözer, <digits>_<orijinal_ad>.png
şeklinde yeniden adlandırır.

Kullanım
--------
# İnternet + Hugging Face ağırlıkları
python digit_agent.py ./captchas

# Çevrimdışı (GGUF)
python digit_agent.py ./captchas --gguf models/gemma-3-4b-it-q4_K_M.gguf

Ek seçenekler
-------------
--digits N   : Beklenen rakam uzunluğu (örn. 6)  
--sleep  s   : Sorgular arası bekleme süresi (varsayılan 0.2 s)
"""

from __future__ import annotations
import argparse, re, time, logging, shutil, os
from pathlib import Path
from typing import Optional
from PIL import Image    # noqa : import zorunlu (vision modları bazen kontrol ediyor)

PROMPT = "<start_of_image> Bu görseldeki rakamları sırasıyla ve eksiksiz yaz."

logging.basicConfig(format="%(levelname)s  %(message)s", level=logging.INFO)

# ─────────────────────  HF transformers yolu (FP16/BF16)  ───────────────────── #

def _hf_solver(img_path: Path, prompt: str) -> str:
    from transformers import pipeline
    import torch

    pipe = _hf_solver._pipe
    if pipe is None:                                    # ilk çağrıda modeli indir/yükle
        pipe = pipeline(
            "image-text-to-text",
            model="google/gemma-3-4b-it",
            cache_dir="./models",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        _hf_solver._pipe = pipe
    return pipe(str(img_path), text=prompt, max_new_tokens=10)[0]["generated_text"]

_hf_solver._pipe = None  # type: ignore


# ─────────────────────────────  GGUF / llama.cpp yolu  ───────────────────────── #

class LlamaVision:
    """GGUF quant ağırlıkları için basit vision arayüzü."""

    def __init__(self, model_path: str):
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=os.cpu_count() or 8,
            n_gpu_layers=35,   # GPU katmanı (VRAM yeterliyse)
        )

    def __call__(self, img_path: Path, prompt: str) -> str:
        with open(img_path, "rb") as f:
            out = self.llm.create_completion(
                prompt=prompt,
                images=[f.read()],
                temperature=0.2,
                max_tokens=8,
            )
        return out["choices"][0]["text"]


# ─────────────────────────────  Yardımcı fonksiyonlar  ───────────────────────── #

_rx_digits = re.compile(r"\d")

def extract_digits(text: str) -> str:
    return "".join(_rx_digits.findall(text))

def rename(img: Path, digits: str) -> Path:
    new_path = img.with_name(f"{digits}_{img.name}")
    shutil.move(img, new_path)
    return new_path


# ────────────────────────────────  Ana iş akışı  ────────────────────────────── #

def run(folder: Path, gguf: Optional[Path], expect_len: int, pause: float):
    solver = (
        LlamaVision(str(gguf))
        if gguf
        else lambda p, q: _hf_solver(p, q)  # type: ignore
    )

    logging.info("🚀 DigitAgent başlıyor – mod: %s", "GGUF" if gguf else "HF pipeline")

    for img in sorted(folder.glob("*.png")):
        logging.info("🔍 %s inceleniyor…", img.name)
        reply = solver(img, PROMPT)
        digits = extract_digits(reply)

        if expect_len and len(digits) != expect_len:
            logging.warning("❌ '%s' %d hane değil, atlandı.", reply, expect_len)
            continue
        if not digits:
            logging.warning("⚠️  Rakam bulunamadı, atlandı.")
            continue

        new_path = rename(img, digits)
        logging.info("✅ %s → %s", img.name, new_path.name)
        time.sleep(pause)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DigitAgent – rakam çözücü/yeniden adlandırıcı")
    ap.add_argument("folder", type=Path, help="PNG dosyalarının bulunduğu klasör")
    ap.add_argument("--gguf", type=Path, help="GGUF modeli (çevrimdışı kullanım)")
    ap.add_argument("--digits", type=int, default=0, help="Beklenen rakam uzunluğu")
    ap.add_argument("--sleep", type=float, default=0.2, help="Sorgular arası bekleme süresi (s)")
    args = ap.parse_args()
    run(args.folder, args.gguf, args.digits, args.sleep)
