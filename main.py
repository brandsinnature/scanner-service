from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import base64
import cv2
import numpy as np
from datetime import datetime
import logging
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Scanner API",
    description="API for processing scanned images and barcodes",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class ScanRequest(BaseModel):
    frame: str  # Base64 encoded image

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

class ScanResponse(BaseModel):
    success: bool
    data: Optional[str] = None
    error: Optional[str] = None

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify API status"""
    try:
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/scan", response_model=ScanResponse)
async def process_scan(request: ScanRequest):
    """Process a scanned image and return detected information"""
    try:
        # Decode base64 image
        image_data = request.frame.split(',')[1] if ',' in request.frame else request.frame
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image")

        # Here you would add your actual image processing logic
        # For example:
        # - Barcode detection
        # - QR code scanning
        # - Object detection
        # This is a placeholder that returns a dummy result
        detected_data = "sample-barcode-123"  # Replace with actual detection logic

        return ScanResponse(
            success=True,
            data=detected_data,
            error=None
        )

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        return ScanResponse(
            success=False,
            error=f"Invalid input: {str(ve)}"
        )
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return ScanResponse(
            success=False,
            error="Failed to process image"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )