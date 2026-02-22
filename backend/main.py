from backend.database import engine, SessionLocal, Base
from backend.models import Prediction
import datetime
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import tempfile
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tables if not exist
Base.metadata.create_all(bind=engine)

# Load model once at startup
model = YOLO("best.pt")


def get_severity(percent):
    if percent < 5:
        return "Mild"
    elif percent < 15:
        return "Moderate"
    else:
        return "Severe"


@app.get("/")
def home():
    return {"message": "Maize Disease API running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(contents)
        temp_path = tmp.name

    results = model(temp_path, conf=0.1)

    os.remove(temp_path)

    # Default values
    disease_name = "Healthy"
    severity_percent = 0
    severity_level = "Healthy"

    # If detection exists
    if results[0].boxes is not None and len(results[0].boxes) > 0:

        cls_id = int(results[0].boxes.cls[0])
        disease_name = model.names[cls_id]

        if disease_name.lower() != "healthy":

            if results[0].masks is not None:
                mask = results[0].masks.data[0].cpu().numpy()
                disease_mask = (mask > 0.5).astype(np.uint8)

                image = Image.open(io.BytesIO(contents)).convert("RGB")
                img = np.array(image)

                leaf_area = img.shape[0] * img.shape[1]
                disease_area = np.sum(disease_mask)

                severity_percent = float((disease_area / leaf_area) * 100)
                severity_level = get_severity(severity_percent)

    # -------- SAVE TO DATABASE --------
    db = SessionLocal()

    new_entry = Prediction(
        disease=disease_name,
        severity_percent=round(severity_percent, 2),
        severity_level=severity_level,
        timestamp=str(datetime.datetime.now())
    )

    db.add(new_entry)
    db.commit()
    db.close()
    # -----------------------------------

    return {
        "disease": disease_name,
        "severity_percent": round(severity_percent, 2),
        "severity_level": severity_level
    }


@app.get("/history")
def get_history():
    db = SessionLocal()
    data = db.query(Prediction).all()
    db.close()
    return data