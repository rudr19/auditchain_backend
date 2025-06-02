from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import shutil
from pathlib import Path
import pandas as pd
from typing import Dict, Any
import uvicorn

from ocr_parser import OCRParser
from anomaly_detector import AnomalyDetector

# Initialize FastAPI app
app = FastAPI(
    title="AuditChain Invoice Fraud Detection API",
    description="API for detecting fraudulent invoices using OCR and machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
ocr_parser = OCRParser()
anomaly_detector = AnomalyDetector()

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("models", exist_ok=True)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "AuditChain Invoice Fraud Detection API is running!"}

@app.post("/upload")
async def upload_invoice(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload and process an invoice file for fraud detection
    
    Args:
        file: Uploaded invoice file (PDF or image)
        
    Returns:
        JSON response with extracted fields and fraud detection results
    """
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_extension = Path(file.filename).suffix.lower()
        allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Save uploaded file
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract text using OCR
        extracted_text = ocr_parser.extract_text(file_path)
        
        # Parse invoice fields
        invoice_fields = ocr_parser.parse_invoice_fields(extracted_text)
        
        # Perform fraud detection
        fraud_result = anomaly_detector.predict_fraud(invoice_fields)
        
        # Combine results
        response = {
            **invoice_fields,
            **fraud_result
        }
        
        # Clean up uploaded file (optional)
        # os.remove(file_path)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/train")
async def train_model(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Train the fraud detection model with historical invoice data
    
    Args:
        file: CSV file with historical invoice data
        
    Returns:
        Training status message
    """
    try:
        # Validate file type
        if not file.filename or not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Please upload a CSV file")
        
        # Save uploaded CSV
        csv_path = f"uploads/training_data.csv"
        with open(csv_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load and validate CSV data
        df = pd.read_csv(csv_path)
        
        # Train the model
        result = anomaly_detector.train_model(df)
        
        # Clean up CSV file
        os.remove(csv_path)
        
        return {"status": "success", "message": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.get("/model-info")
async def model_info() -> Dict[str, Any]:
    """Get information about the current model"""
    try:
        info = anomaly_detector.get_model_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
