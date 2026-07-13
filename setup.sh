#!/bin/bash
set -e

echo "=== LocateAnything Full Setup ==="

# 1. Install dependencies
pip install hf_transfer opencv-python-headless==4.11.0.86 transformers==4.57.1 numpy==1.26.4 Pillow==11.1.0 peft decord==0.6.0 lmdb==1.7.5 fastapi 'uvicorn[standard]' python-multipart websockets

# 2. Download the model (skips if already present on persistent volume)
if [ ! -d "/workspace/locate-anything-3b" ]; then
    echo "Downloading model..."
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='nvidia/LocateAnything-3B', local_dir='/workspace/locate-anything-3b', ignore_patterns=['assets/*.mp4','assets/*.png','assets/*.jpg','*.png'])"
else
    echo "Model already present, skipping download."
fi

# 3. Write server.py
cat > /workspace/server.py << 'PYEOF'
import re
import torch
import io
import json
import asyncio
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import base64

app = FastAPI()

model_path = "/workspace/locate-anything-3b"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda").eval()
print("Model ready!")


def parse_output(answer, w, h):
    """Parse model output into pixel-coordinate boxes and points."""
    boxes = []
    points = []
    box_pattern = re.compile(r"(?:<ref>([^<]+)</ref>)?<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
    point_pattern = re.compile(r"(?:<ref>([^<]+)</ref>)?<box><(\d+)><(\d+)></box>(?!<)")

    for m in box_pattern.finditer(answer):
        label, x1, y1, x2, y2 = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
        boxes.append({
            "label": label,
            "x1": x1 / 1000 * w,
            "y1": y1 / 1000 * h,
            "x2": x2 / 1000 * w,
            "y2": y2 / 1000 * h,
        })

    for m in point_pattern.finditer(answer):
        label, x, y = m.group(1), int(m.group(2)), int(m.group(3))
        points.append({
            "label": label,
            "x": x / 1000 * w,
            "y": y / 1000 * h,
        })

    return boxes, points


def run_inference(img: Image.Image, prompt: str, mode: str = "hybrid", max_new_tokens: int = 2048):
    w, h = img.size

    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt},
    ]}]

    text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = processor.process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")

    with torch.no_grad():
        response = model.generate(
            pixel_values=inputs["pixel_values"].to(torch.bfloat16),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws"),
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=mode,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    answer = response[0] if isinstance(response, tuple) else response
    boxes, points = parse_output(answer, w, h)

    return {
        "raw": answer,
        "boxes": boxes,
        "points": points,
        "image_size": {"w": w, "h": h},
        "prompt": prompt,
    }


def build_prompt(task: str, phrase: str = None, categories: str = None):
    """Build the correct prompt template for each task."""
    if task == "detect":
        cats = "</c>".join([c.strip() for c in categories.split(",")])
        return f"Locate all the instances that matches the following description: {cats}."
    elif task == "ground_single":
        return f"Locate a single instance that matches the following description: {phrase}."
    elif task == "ground_multi":
        return f"Locate all the instances that match the following description: {phrase}."
    elif task == "ground_text":
        return f"Please locate the text referred as {phrase}."
    elif task == "detect_text":
        return "Detect all the text in box format."
    elif task == "ground_gui_box":
        return f"Locate the region that matches the following description: {phrase}."
    elif task == "ground_gui_point":
        return f"Point to: {phrase}."
    elif task == "point":
        return f"Point to: {phrase}."
    return phrase


async def load_image(image: UploadFile) -> Image.Image:
    img_bytes = await image.read()
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


# ---- HTTP endpoints ----

@app.post("/detect")
async def http_detect(
    image: UploadFile = File(...),
    categories: str = Form(...),
    mode: str = Form("hybrid"),
):
    """Object detection / document layout analysis. Pass comma-separated categories."""
    img = await load_image(image)
    prompt = build_prompt("detect", categories=categories)
    return JSONResponse(run_inference(img, prompt, mode))


@app.post("/ground_single")
async def http_ground_single(
    image: UploadFile = File(...),
    phrase: str = Form(...),
    mode: str = Form("hybrid"),
):
    """Phrase grounding — single instance only."""
    img = await load_image(image)
    prompt = build_prompt("ground_single", phrase=phrase)
    return JSONResponse(run_inference(img, prompt, mode))


@app.post("/ground_multi")
async def http_ground_multi(
    image: UploadFile = File(...),
    phrase: str = Form(...),
    mode: str = Form("hybrid"),
):
    """Phrase grounding — multiple instances."""
    img = await load_image(image)
    prompt = build_prompt("ground_multi", phrase=phrase)
    return JSONResponse(run_inference(img, prompt, mode))


@app.post("/ground_text")
async def http_ground_text(
    image: UploadFile = File(...),
    phrase: str = Form(...),
    mode: str = Form("hybrid"),
):
    """Find specific text in an image."""
    img = await load_image(image)
    prompt = build_prompt("ground_text", phrase=phrase)
    return JSONResponse(run_inference(img, prompt, mode))


@app.post("/detect_text")
async def http_detect_text(
    image: UploadFile = File(...),
    mode: str = Form("hybrid"),
):
    """Scene text detection — find all text."""
    img = await load_image(image)
    prompt = build_prompt("detect_text")
    return JSONResponse(run_inference(img, prompt, mode))


@app.post("/ground_gui")
async def http_ground_gui(
    image: UploadFile = File(...),
    phrase: str = Form(...),
    output_type: str = Form("box"),
    mode: str = Form("hybrid"),
):
    """GUI grounding — for buttons, menus, UI elements. output_type: box | point"""
    img = await load_image(image)
    task = "ground_gui_point" if output_type == "point" else "ground_gui_box"
    prompt = build_prompt(task, phrase=phrase)
    return JSONResponse(run_inference(img, prompt, mode))


@app.post("/point")
async def http_point(
    image: UploadFile = File(...),
    phrase: str = Form(...),
    mode: str = Form("hybrid"),
):
    """Pointing — return a single point coordinate."""
    img = await load_image(image)
    prompt = build_prompt("point", phrase=phrase)
    return JSONResponse(run_inference(img, prompt, mode))


@app.post("/locate")
async def http_locate(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    mode: str = Form("hybrid"),
):
    """Generic endpoint — pass a raw prompt yourself."""
    img = await load_image(image)
    return JSONResponse(run_inference(img, prompt, mode))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "endpoints": {
            "/detect": "Object detection (categories: 'person, car, dog')",
            "/ground_single": "Single instance grounding (phrase)",
            "/ground_multi": "Multiple instance grounding (phrase)",
            "/ground_text": "Find specific text (phrase)",
            "/detect_text": "Scene text detection (all text)",
            "/ground_gui": "GUI element grounding (phrase, output_type: box|point)",
            "/point": "Pointing — single point (phrase)",
            "/locate": "Generic — raw prompt",
            "/ws": "WebSocket — persistent connection, all tasks",
            "/health": "Health check",
        }
    }


# ---- WebSocket endpoint (persistent connection, all tasks) ----

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Send JSON messages of the form:
    {
        "task": "detect" | "ground_single" | "ground_multi" | "ground_text" |
                "detect_text" | "ground_gui_box" | "ground_gui_point" | "point" | "locate",
        "phrase": "...",       # required for most tasks
        "categories": "...",   # required only for "detect"
        "prompt": "...",       # required only for "locate"
        "mode": "hybrid",      # optional, default hybrid
        "image": "<base64 jpeg/png>"
    }
    """
    await websocket.accept()
    print("WebSocket client connected")
    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            task = msg.get("task")
            phrase = msg.get("phrase")
            categories = msg.get("categories")
            mode = msg.get("mode", "hybrid")
            img_b64 = msg.get("image")

            img_bytes = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            if task == "locate":
                prompt = msg.get("prompt", phrase)
            elif task == "detect":
                prompt = build_prompt("detect", categories=categories)
            else:
                prompt = build_prompt(task, phrase=phrase)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_inference, img, prompt, mode)

            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
            await websocket.close()
        except Exception:
            pass
PYEOF

echo "=== Setup complete. ==="
echo "To start the server, run:"
echo "cd /workspace && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uvicorn server:app --host 0.0.0.0 --port 8000"
