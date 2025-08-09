# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import disease, yield_pred

app = FastAPI(
    title="Cassava AI API",
    description="Disease detection + Yield prediction",
    version="1.0.0"
)

origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    # add your deployed frontends here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(disease.router)
# app.include_router(yield_pred.router)

@app.get("/")
def root():
    return {"message": "Cassava AI API running", "routes": ["/disease/predict", "/yield/predict"]}
