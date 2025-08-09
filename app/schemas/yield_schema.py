from pydantic import BaseModel, Field
from typing import Literal

class YieldInput(BaseModel):
    plant_height_cm: float = Field(..., ge=0)
    stem_diameter_cm: float = Field(..., ge=0)
    leaf_count: int = Field(..., ge=0)
    plant_age_months: int = Field(..., ge=0)
    soil_moisture: Literal["Low", "Medium", "High"]
    avg_temp_c: float
    fertilizer_application: Literal["None", "Organic", "Inorganic", "Mixed"]
    cassava_variety: str
    planting_density: int = Field(..., ge=1)

class YieldOutput(BaseModel):
    predicted_yield_tph: float
