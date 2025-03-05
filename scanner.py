import openai_func
from upi_scanner import extract_upi_info

import cv2
import re
import numpy as np
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_objects(frame, client):
    products = openai_func.detect_object_openai(client, frame)
    
    result = {
        "detections": products,
    }

    return result

def get_upi(frame):
    # Check if frame is a base64 string or URL
    if isinstance(frame, str):
        logger.info("Processing frame as a string...")
        if frame.startswith('data:image'):
            logger.info("Processing frame as a data URI...")
            # Extract base64 data from data URI
            # Find the base64 part after the comma
            base64_data = frame.split(',')[1]
            # Decode base64 to bytes
            img_bytes = base64.b64decode(base64_data)
            # Convert to numpy array
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            # Decode the image
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        elif frame.startswith('http://') or frame.startswith('https://'):
            # Download image from URL
            import urllib.request
            
            resp = urllib.request.urlopen(frame)
            img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
    return extract_upi_info(frame)
    