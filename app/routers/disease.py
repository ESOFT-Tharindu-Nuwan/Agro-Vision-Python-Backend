from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io, numpy as np, tensorflow as tf
from typing import Optional

router = APIRouter(prefix="/disease", tags=["Disease"])

MODEL_PATH = "app/models/sri_lankan_cassava_model_20250627_021853.h5"
class_names = ['CMD', 'BLS', 'CBB', 'Healthy']
IMG_HEIGHT = 224
IMG_WIDTH = 224
CONFIDENCE_THRESHOLD = 0.75

_model: Optional[tf.keras.Model] = None

def get_model() -> tf.keras.Model:
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Disease model loaded: {MODEL_PATH}")
    return _model

@router.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image (JPG/PNG).")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        image_array = np.array(image, dtype=np.float32)[None, ...] / 255.0
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image error: {e}")

    model = get_model()
    preds = model.predict(image_array)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx])
    name = class_names[idx]

    if conf < CONFIDENCE_THRESHOLD:
        return JSONResponse({
            "status": "unidentifiable",
            "message": f"Low confidence. Highest: {name} ({conf:.2f}).",
            "predicted_class": name,
            "confidence": conf,
            "all_probabilities": {n: float(p) for n, p in zip(class_names, preds)}
        })

    return JSONResponse({
        "status": "success",
        "message": "Disease identified successfully.",
        "predicted_class": name,
        "confidence": conf,
        "all_probabilities": {n: float(p) for n, p in zip(class_names, preds)}
    })
