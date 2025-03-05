import openai_func
import cv2
import re
import numpy as np
import base64

def detect_objects(frame, client):
    products = openai_func.detect_object_openai(client, frame)
    
    result = {
        "detections": products,
    }

    return result

def get_upi(frame):
        # Decode the QR code using OpenCV's QR code detector
        qr_detector = cv2.QRCodeDetector()
        decoded_objects = []
        
        # Check if frame is a base64 string or URL
        if isinstance(frame, str):
            if frame.startswith('data:image'):
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
        
        retval, decoded_info, points, straight_qrcode = qr_detector.detectAndDecodeMulti(frame)
        
        if retval:
            for info in decoded_info:
                if info:  # Check if the information is not empty
                    decoded_objects.append(type('obj', (), {'data': info.encode('utf-8')}))
        
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
        return {"upi_id": None, "name": None}