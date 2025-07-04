#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DigitAgent v2 – number-CAPTCHA renaming agent
============================================

Özellikler
----------
• Kaynak klasördeki PNG'leri tarar
• Gemma-3 4B IT modeline “What are the numbers?” sorusunu sorar
• Yanıttaki JSON'dan `digits` alanını alır
• Hedef klasöre <digits>.png adıyla kopyalar (collision → _1, _2 …)
• HF pipeline (internet) veya GGUF (tam çevrimdışı) ile çalışır

Kullanım
--------
python digit_agent.py <src_dir> <dst_dir>             # HF yolu
python digit_agent.py <src_dir> <dst_dir> --gguf models/gemma-3-4b-it-Q4_K_M.gguf
"""

from __future__ import annotations
import argparse, json, logging, os, re, shutil, time
from pathlib import Path
from typing import Optional
from PIL import Image   # noqa: pillow vision modüllerini tetikler

PROMPT = (
    "<start_of_image>\nYou are a helpful vision agent.\n"
    "Task: Identify the digits in this image.\n"
    'Respond ONLY in JSON like {"digits": "123456"}'
)

logging.basicConfig(format="%(levelname)s  %(message)s", level=logging.INFO)

# ─────────────────────────── HF transformers yolu ────────────────────────── #

def _hf_solver(img_path: Path) -> str:
    from transformers import pipeline
    import torch

    pipe = _hf_solver._pipe
    if pipe is None:                                        # lazy-load
        pipe = pipeline(
            "image-text-to-text",
            model="google/gemma-3-4b-it",
            cache_dir="./models",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        _hf_solver._pipe = pipe
    return pipe(str(img_path), text=PROMPT, max_new_tokens=20)[0]["generated_text"]

_hf_solver._pipe = None  # type: ignore

# ───────────────────────────── GGUF / llama.cpp yolu ─────────────────────── #

class LlamaVision:
    def __init__(self, model_path: str):
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=os.cpu_count() or 8,
            n_gpu_layers=35,
        )

    def __call__(self, img_path: Path) -> str:
        with open(img_path, "rb") as f:
            ans = self.llm.create_completion(
                prompt=PROMPT,
                images=[f.read()],
                temperature=0.1,
                max_tokens=20,
            )
        return ans["choices"][0]["text"]

# ───────────────────────────── Yardımcı fonksiyonlar ─────────────────────── #

_digits_rx = re.compile(r'\"digits\"\s*:\s*\"(\d+)\"')

def parse_digits(response: str) -> Optional[str]:
    # 1) JSON parselemeyi dene
    try:
        data = json.loads(response)
        if isinstance(data, dict) and "digits" in data:
            return str(data["digits"])
    except json.JSONDecodeError:
        pass
    # 2) Regex yedeği
    m = _digits_rx.search(response)
    return m.group(1) if m else None

def safe_copy(src: Path, dst_dir: Path, digits: str) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    base = digits
    counter = 0
    while True:
        new_name = f"{base}{f'_{counter}' if counter else ''}{src.suffix}"
        dst_path = dst_dir / new_name
        if not dst_path.exists():
            shutil.copy2(src, dst_path)
            return dst_path
        counter += 1

# ──────────────────────────────────  Ana akış  ───────────────────────────── #

def run(src_dir: Path, dst_dir: Path, gguf: Optional[Path], pause: float):
    solver = (
        LlamaVision(str(gguf)) if gguf else lambda p: _hf_solver(p)  # type: ignore
    )

    logging.info("🚀 DigitAgent başlıyor – mod: %s",
                 f"GGUF ({gguf})" if gguf else "HF pipeline")

    for img in sorted(src_dir.glob("*.png")):
        logging.info("🔍 %s işleniyor…", img.name)
        reply = solver(img).strip()
        digits = parse_digits(reply)

        if not digits:
            logging.warning("⚠️  Rakam bulunamadı → yanıt: %s", reply)
            continue

        copied = safe_copy(img, dst_dir, digits)
        logging.info("✅ %s → %s", img.name, copied.name)
        time.sleep(pause)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DigitAgent v2")
    ap.add_argument("src", type=Path, help="Kaynak klasör (PNG'ler)")
    ap.add_argument("dst", type=Path, help="Hedef klasör (yeniden adlandırılmış)")
    ap.add_argument("--gguf", type=Path, help="GGUF modeli (çevrimdışı kullanım)")
    ap.add_argument("--sleep", type=float, default=0.2, help="Sorgular arası bekleme")
    args = ap.parse_args()
    run(args.src, args.dst, args.gguf, args.sleep)
