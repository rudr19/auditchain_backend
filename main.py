from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
import shutil
import pandas as pd
from typing import Dict, Any
import uvicorn

from ocr_parser import OCRParser
from anomaly_detector import AnomalyDetector

app = FastAPI(
    title="AuditChain Invoice Fraud Detection API",
    description="API for detecting fraudulent invoices using OCR and machine learning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ocr_parser = OCRParser()
anomaly_detector = AnomalyDetector()

os.makedirs("uploads", exist_ok=True)
os.makedirs("models", exist_ok=True)

@app.get("/")
async def root():
    return {"message": "AuditChain Invoice Fraud Detection API is running!"}

@app.post("/upload")
async def upload_invoice(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        file_extension = Path(file.filename).suffix.lower()
        allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )

        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extracted_text = ocr_parser.extract_text(file_path)
        invoice_fields = ocr_parser.parse_invoice_fields(extracted_text)
        fraud_result = anomaly_detector.predict_fraud(invoice_fields)

        response = {**invoice_fields, **fraud_result}

        # Optional: clean up uploaded file
        # os.remove(file_path)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/train")
async def train_model(file: UploadFile = File(...)) -> Dict[str, str]:
    try:
        if not file.filename or not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Please upload a CSV file")

        csv_path = "uploads/training_data.csv"
        with open(csv_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        df = pd.read_csv(csv_path)
        result = anomaly_detector.train_model(df)

        os.remove(csv_path)

        return {"status": "success", "message": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.get("/model-info")
async def model_info() -> Dict[str, Any]:
    try:
        info = anomaly_detector.get_model_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

