from fastapi import APIRouter, HTTPException
# from app.schemas.yield import YieldInput, YieldOutput
import joblib, pandas as pd
from typing import Optional

router = APIRouter(prefix="/yield", tags=["Yield"])

MODEL_PATH = "app/models/cassava_yield_pipeline.joblib"
_yield_model: Optional[object] = None

def get_yield_model():
    global _yield_model
    if _yield_model is None:
        try:
            _yield_model = joblib.load(MODEL_PATH)
            print(f"✅ Yield pipeline loaded: {MODEL_PATH}")
        except Exception as e:
            raise RuntimeError(f"Could not load yield model: {e}")
    return _yield_model

@router.post("/predict", response_model=YieldOutput)
def predict_yield(payload: YieldInput):
    model = get_yield_model()
    try:
        df = pd.DataFrame([payload.dict()])
        pred = float(model.predict(df)[0])
        return YieldOutput(predicted_yield_tph=pred)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {e}")
