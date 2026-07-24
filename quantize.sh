#!/usr/bin/env bash
# LocateAnything-3B quantized server setup — RunPod (RTX 2000 Ada / 16GB target)
# Usage: curl -sL https://raw.githubusercontent.com/AmeerUmar320/server-setup/main/setup.sh | bash
set -euo pipefail

# ---- config (override via env before piping, e.g. VRAM_GIB=4 curl ... | bash) ----
WORKDIR="${WORKDIR:-/workspace}"
MODEL_REPO="${MODEL_REPO:-nvidia/LocateAnything-3B}"
MODEL_DIR="${WORKDIR}/locateanything-3b"
VENV_DIR="${WORKDIR}/venv"
MEM_FRACTION="${MEM_FRACTION:-0.5}" # torch.cuda.set_per_process_memory_fraction
QUANTIZE="${QUANTIZE:-1}"           # 1 = 4-bit nf4 on LM only, 0 = bf16 baseline
PORT="${PORT:-8000}"

echo "== LocateAnything-3B setup =="
echo "workdir=${WORKDIR} vram_gib=${VRAM_GIB} mem_fraction=${MEM_FRACTION} quantize=${QUANTIZE} port=${PORT}"

mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

# ---- system deps ----
apt-get update -qq
apt-get install -y -qq git git-lfs python3-venv >/dev/null
git lfs install --skip-repo

# ---- venv ----
if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip -q

# torch/cuda already in the runpod image — only install what's missing
pip install -q transformers accelerate bitsandbytes fastapi "uvicorn[standard]" \
    websockets pillow einops timm huggingface_hub

# ---- pull model onto the volume, not the container disk ----
if [ ! -d "${MODEL_DIR}" ] || [ -z "$(ls -A "${MODEL_DIR}" 2>/dev/null)" ]; then
  echo "== downloading ${MODEL_REPO} =="
  huggingface-cli download "${MODEL_REPO}" --local-dir "${MODEL_DIR}"
else
  echo "== model already present at ${MODEL_DIR}, skipping download =="
fi

# ---- write inspect_modules.py (run once manually to confirm skip-module names) ----
cat > "${WORKDIR}/inspect_modules.py" <<'PYEOF'
from transformers import AutoModel
import sys

model_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace/locateanything-3b"
m = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
print("Top-level modules (use these in SKIP_MODULES if they differ from defaults):")
for n, _ in m.named_children():
    print(" -", n)
PYEOF

# ---- write model_loader.py ----
cat > "${WORKDIR}/model_loader.py" <<'PYEOF'
import os
import torch
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

MODEL_PATH = os.environ.get("MODEL_DIR", "/workspace/locateanything-3b")

# Confirm these against inspect_modules.py output before trusting them.
# Never quantize the vision tower or the box-decoding head — grounding
# precision lives there, and it degrades silently (text output still
# looks fine while click accuracy quietly drops).
SKIP_MODULES = os.environ.get(
    "SKIP_MODULES", "vision_model,mlp1,box_head,lm_head"
).split(",")


def load_model(mem_fraction: float = 1.0, quantize: bool = True):
    torch.cuda.set_per_process_memory_fraction(mem_fraction, device=0)

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    kwargs = dict(
        trust_remote_code=True,
        device_map="cuda:0",  # fixed GPU, no CPU offload possible
    )

    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=SKIP_MODULES,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModel.from_pretrained(MODEL_PATH, **kwargs)
    model.eval()

    alloc = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    print(f"[loaded] allocated={alloc:.2f}GB reserved={reserved:.2f}GB (device=cuda:0, no offload)")

    return model, processor
PYEOF

# ---- write server.py ----
cat > "${WORKDIR}/server.py" <<'PYEOF'
import io
import time
import base64
import json
import argparse

import torch
from fastapi import FastAPI, WebSocket
from PIL import Image

from model_loader import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--mem-fraction", type=float, default=float(__import__("os").environ.get("MEM_FRACTION", 0.5)))
parser.add_argument("--quantize", type=int, default=int(__import__("os").environ.get("QUANTIZE", 1)))
args, _ = parser.parse_known_args()

app = FastAPI()
model, processor = load_model(args.mem_fraction, bool(args.quantize))

# warm-up so the first real request isn't measuring lazy init/compile
_dummy = Image.new("RGB", (1024, 768))
with torch.inference_mode():
    try:
        _inputs = processor(images=_dummy, text="warmup", return_tensors="pt").to("cuda:0")
        model.generate(**_inputs, max_new_tokens=4)
        torch.cuda.reset_peak_memory_stats(0)
    except Exception as e:
        print(f"[warmup] skipped ({e}) — check processor/generate call against the model's actual API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.websocket("/ground")
async def ground(ws: WebSocket):
    await ws.accept()
    while True:
        payload = json.loads(await ws.receive_text())
        image = Image.open(io.BytesIO(base64.b64decode(payload["image"]))).convert("RGB")
        prompt = payload["prompt"]

        t0 = time.perf_counter()
        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=64)
        result = processor.decode(out[0], skip_special_tokens=True)
        latency_ms = (time.perf_counter() - t0) * 1000

        peak_mb = torch.cuda.max_memory_allocated(0) / 1e6
        await ws.send_text(json.dumps({
            "result": result,
            "latency_ms": round(latency_ms, 1),
            "peak_vram_mb": round(peak_mb, 1),
        }))
PYEOF

# ---- write a bench client stub ----
cat > "${WORKDIR}/bench.py" <<'PYEOF'
"""
Sweep helper — point this at your own labeled screenshots to find the
real floor (load floor vs run floor differ; run floor is always higher
because of activation memory during ViT prefill).

TEST_SET must be a list of (image_path, prompt) tuples you define.
"""
import asyncio
import base64
import json
import statistics

import websockets

TEST_SET = [
    # ("path/to/screenshot1.png", "the Save button"),
]


async def run(uri="ws://localhost:8000/ground"):
    if not TEST_SET:
        print("Populate TEST_SET with (image_path, prompt) tuples first.")
        return
    latencies, peaks = [], []
    async with websockets.connect(uri) as ws:
        for img_path, prompt in TEST_SET:
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            await ws.send(json.dumps({"image": b64, "prompt": prompt}))
            resp = json.loads(await ws.recv())
            latencies.append(resp["latency_ms"])
            peaks.append(resp["peak_vram_mb"])
            print(resp)

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95)]
    print(f"\np50={p50:.0f}ms p95={p95:.0f}ms peak_vram_max={max(peaks):.0f}MB")


if __name__ == "__main__":
    asyncio.run(run())
PYEOF

echo ""
echo "== setup complete =="
echo "Next steps:"
echo "  1. source ${VENV_DIR}/bin/activate"
echo "  2. python ${WORKDIR}/inspect_modules.py   # confirm real module names, set SKIP_MODULES env if they differ"
echo "  3. cd ${WORKDIR} && MEM_FRACTION=${MEM_FRACTION} QUANTIZE=${QUANTIZE} \\"
echo "       uvicorn server:app --host 0.0.0.0 --port ${PORT}"
echo "  4. fill in TEST_SET in ${WORKDIR}/bench.py, then: python ${WORKDIR}/bench.py"
echo ""
echo "To sweep VRAM floor: rerun step 3 with different VRAM_GIB / MEM_FRACTION,"
echo "watch [loaded] device_map output and peak_vram_mb in responses."
