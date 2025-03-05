import openai_func
import cv2
import re
import numpy as np
import base64
import zxing
import logging
import tempfile
import PIL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_objects(frame, client):
    products = openai_func.detect_object_openai(client, frame)
    
    result = {
        "detections": products,
    }

    return result

def extract_upi_details(qr_code_data):
    """Extract UPI ID and Name from QR code data."""
    upi_match = re.search(r'pa=([\w.@]+)', qr_code_data)
    name_match = re.search(r'pn=([^&]+)', qr_code_data)

    upi_id = upi_match.group(1) if upi_match else "Not found"
    name = name_match.group(1) if name_match else "Not found"

    return upi_id, name

def scan_qr_from_image(image):
    """Scan QR code from an image object (PIL Image) and extract UPI details."""
    # Save image temporarily
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_file:
        image.save(temp_file.name)  # Save PIL image
        reader = zxing.BarCodeReader()
        barcode = reader.decode(temp_file.name)

    if barcode and barcode.raw:
        upi_id, name = extract_upi_details(barcode.raw)
        
        return {
            "upi_id": upi_id,
            "name": name
        }
    else:
        return {"upi_id": None, "name": None}

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
        
        return scan_qr_from_image(PIL.Image.fromarray(frame))
    