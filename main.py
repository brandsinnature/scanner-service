import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
from datetime import datetime
import logging
from scanner import detect_objects
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
    logger.info("Processing scan request...")
    try:
        # Simulate detection
        detection_result = detect_objects(request.frame)

        # Return the detection result
        return ScanResponse(
            success=True,
            data=detection_result,
            error=None
        )
    except Exception as e:
        logger.error(f"Error processing scan request: {str(e)}")
        return ScanResponse(
            success=False,
            data=None,
            error="Failed to process scan request."
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )