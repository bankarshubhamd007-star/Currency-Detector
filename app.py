import io
import torch
import logging
from typing import Optional
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from transformers import AutoModelForImageClassification, AutoImageProcessor
import requests
import numpy as np
from torchvision import transforms

import os
import uuid
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEEDBACK_DIR = Path("feedback_dataset")
TEMP_IMAGE_DIR = Path("temp_uploads")
for label in ["Real", "Fake"]:
    (FEEDBACK_DIR / label).mkdir(parents=True, exist_ok=True)
TEMP_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Indian Currency Detection", version="1.0.0")

# Model configuration
HF_MODEL_ID = "Shubhamm007/indian-currency-classifier"
LOCAL_MODEL_PATH = "./improved_model"
MODEL_ID = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else HF_MODEL_ID

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables
model = None
image_processor = None

# Manual Preprocessing (Standard for EfficientNetV2-S)
manual_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def save_temp_image(img_bytes: bytes) -> str:
    temp_id = str(uuid.uuid4())
    (TEMP_IMAGE_DIR / f"{temp_id}.img").write_bytes(img_bytes)
    return temp_id


def load_image_bytes(image_temp_id: Optional[str] = None, image_url: Optional[str] = None) -> bytes:
    if image_temp_id:
        temp_path = TEMP_IMAGE_DIR / f"{image_temp_id}.img"
        if not temp_path.exists():
            raise HTTPException(status_code=404, detail="Temporary image not found")
        return temp_path.read_bytes()

    if image_url:
        resp = requests.get(image_url, timeout=10)
        resp.raise_for_status()
        return resp.content

    raise HTTPException(status_code=400, detail="No image provided")


def delete_temp_image(image_temp_id: Optional[str]) -> None:
    if not image_temp_id:
        return

    temp_path = TEMP_IMAGE_DIR / f"{image_temp_id}.img"
    if temp_path.exists():
        temp_path.unlink()

def load_model():
    global model, image_processor
    if model is not None: return True
    try:
        logger.info(f"Loading model: {MODEL_ID}")
        # Load Model
        model = AutoModelForImageClassification.from_pretrained(MODEL_ID, trust_remote_code=True)
        model.to(device).eval()
        
        # Load Processor from YOUR repo (since you fixed it!)
        try:
            image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
            logger.info("✓ Model and Processor loaded from your repository.")
        except Exception as pe:
            logger.warning(f"⚠ Could not load processor from repo: {pe}. Using manual fallback.")
            image_processor = None
            
        return True
    except Exception as e:
        logger.error(f"Load error: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/", response_class=HTMLResponse)
async def get_index():
    template_path = Path(__file__).parent / "templates" / "index.html"
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None)
):
    if model is None: load_model()
    
    try:
        if file:
            img_bytes = await file.read()
        elif image_url:
            img_bytes = load_image_bytes(image_url=image_url)
        else:
            raise HTTPException(status_code=400, detail="No image provided")

        image = Image.open(io.BytesIO(img_bytes))
        if image.mode != "RGB": image = image.convert("RGB")
        image_temp_id = save_temp_image(img_bytes)
        
        # Preprocess
        if image_processor:
            inputs = image_processor(images=image, return_tensors="pt").to(device)
            pixel_values = inputs["pixel_values"]
        else:
            # The bulletproof fallback
            pixel_values = manual_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(pixel_values)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            conf = probs[0][pred_class].item() * 100
            
        labels = {0: "Real Currency", 1: "Fake Currency"}
        return {
            "prediction": labels.get(pred_class, "Unknown"),
            "confidence": round(conf, 2),
            "is_real": pred_class == 0,
            "image_temp_id": image_temp_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def feedback(
    correct_label: str = Form(...),
    image_temp_id: Optional[str] = Form(None),
    image_url: Optional[str] = Form(None)
):
    try:
        if correct_label not in {"Real", "Fake"}:
            raise HTTPException(status_code=400, detail="Invalid feedback label")

        img_content = load_image_bytes(image_temp_id=image_temp_id, image_url=image_url)
        
        # Calculate unique hash (Digital Fingerprint)
        img_hash = hashlib.sha256(img_content).hexdigest()
        file_name = f"{img_hash}.jpg"
        
        # Check if this exact image already exists in ANY feedback folder
        already_exists = False
        for label in ["Real", "Fake"]:
            if (FEEDBACK_DIR / label / file_name).exists():
                already_exists = True
                break
        
        if already_exists:
            logger.info(f"Skipping duplicate feedback image: {img_hash[:8]}")
            delete_temp_image(image_temp_id)
            return {"status": "success", "message": "We already have this image in our database! Thank you."}

        # Save the new unique image
        label_dir = FEEDBACK_DIR / correct_label
        with open(label_dir / file_name, "wb") as f:
            f.write(img_content)
            
        logger.info(f"✓ Saved NEW unique feedback image: {img_hash[:8]}")
        delete_temp_image(image_temp_id)
        return {"status": "success", "message": "Thank you! New unique feedback saved."}
    except HTTPException as e:
        logger.error(f"Feedback error: {e.detail}")
        return {"status": "error", "message": e.detail}
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
