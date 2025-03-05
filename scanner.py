import json
import openai_func
import cv2
from pyzbar.pyzbar import decode
import re

def detect_objects(frame, client):
    products = openai_func.detect_object_openai(client, frame)
    
    result = {
        "detections": products,
    }

    return result

def get_upi(frame):
        # Decode the QR code
        decoded_objects = decode(frame)
        
        for obj in decoded_objects:
            # Get the data from the QR code
            data = obj.data.decode('utf-8')
            
            # Check if it's a UPI QR code
            if data.startswith("upi://"):
                # Extract the UPI ID using regex
                # Extract UPI ID
                upi_match = re.search(r"pa=([^&]+)", data)
                # Extract name (usually provided in the 'pn' parameter)
                name_match = re.search(r"pn=([^&]+)", data)
                return {
                    "upi": upi_match.group(1),
                    "name": name_match.group(1) if name_match else None
                }
        
        # Return None if no UPI ID found
        return {"upi_id": None}