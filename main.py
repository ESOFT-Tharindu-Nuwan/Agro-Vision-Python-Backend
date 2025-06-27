# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import os

# --- 1. Initialize FastAPI App ---
app = FastAPI(
    title="Sri Lankan Cassava Disease Detection API",
    description="API for classifying cassava leaf diseases (CMD, BLS, CBB, Healthy) from images.",
    version="1.0.0"
)

# --- 2. Load the Trained Model (happens once when API starts) ---
# Define the path to your .h5 model file
# IMPORTANT: Update this filename to match your downloaded model's name!
MODEL_PATH = "models/sri_lankan_cassava_model_20250627_021853.h5"

model = None
class_names = ['CMD', 'BLS', 'CBB', 'Healthy']
IMG_HEIGHT = 224
IMG_WIDTH = 224
CONFIDENCE_THRESHOLD = 0.75

@app.on_event("startup")
async def load_model():
    """
    Load the TensorFlow Keras model when the FastAPI application starts up.
    This ensures the model is loaded only once and is available for all requests.
    """
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        model.summary()
        print(f"✅ Model '{MODEL_PATH}' loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise RuntimeError(f"Could not load model at {MODEL_PATH}. Please check the path and file.")

# --- 3. Prediction Endpoint ---
@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """
    Receives an image file, preprocesses it, and predicts the cassava leaf disease.

    Args:
        file (UploadFile): The image file to be classified.

    Returns:
        JSONResponse: A JSON object containing the prediction, confidence,
                      and a message.
    """
    # --- 3.1. Validate File Type ---
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file (e.g., JPG, PNG)."
        )

    # --- 3.2. Read and Preprocess Image ---
    try:
        # Read the image file into a PIL Image object
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Resize image to the target size expected by the model
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))

        # Convert PIL Image to NumPy array and normalize pixel values
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array / 255.0
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing image: {e}. Please ensure it's a valid image."
        )

    # --- 3.3. Make Prediction ---
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please check server logs for startup errors."
        )

    predictions = model.predict(image_array)[0]

    # Get the highest confidence prediction
    predicted_class_index = np.argmax(predictions)
    predicted_confidence = float(predictions[predicted_class_index])
    predicted_class_name = class_names[predicted_class_index]

    # --- 3.4. Handle "Cannot Identify" or "Not a Cassava Leaf" ---
    # This is a heuristic. If the highest confidence is below a threshold,
    # we consider it "unidentifiable" or potentially not a cassava leaf.
    if predicted_confidence < CONFIDENCE_THRESHOLD:
        message = (
            f"Cannot confidently identify the leaf or it might not be a cassava leaf. "
            f"Highest prediction: {predicted_class_name} with {predicted_confidence:.2f} confidence."
        )
        return JSONResponse(content={
            "status": "unidentifiable",
            "message": message,
            "predicted_class": predicted_class_name,
            "confidence": predicted_confidence,
            "all_probabilities": {name: float(prob) for name, prob in zip(class_names, predictions)}
        })

    # --- 3.5. Return Successful Prediction ---
    return JSONResponse(content={
        "status": "success",
        "message": "Disease identified successfully.",
        "predicted_class": predicted_class_name,
        "confidence": predicted_confidence,
        "all_probabilities": {name: float(prob) for name, prob in zip(class_names, predictions)}
    })

# --- Optional: Root Endpoint for Health Check ---
@app.get("/")
async def root():
    """
    Root endpoint for a simple health check.
    """
    return {"message": "Cassava Disease Detection API is running!"}
