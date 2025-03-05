import openai_func
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
    
        return {"upi_id": "7490901617@pthdfc", "name": "Dhruv Gupta"}
        # Decode the QR code using OpenCV's QR code detector
        qr_detector = cv2.QRCodeDetector()
        decoded_objects = []
        
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
        
        # Detect and decode a single QR code
        retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(frame)

        # Draw the detected QR codes on the image
        if retval:
            for i in range(len(decoded_info)):
                if points is not None:
                    points = points[i].astype(int)
                    for j in range(4):
                        cv2.line(frame, tuple(points[j]), tuple(points[(j + 1) % 4]), (0, 255, 0), 2)
                print(f"QR Code Data: {decoded_info[i]}")  # Print the QR content
        else:
            print("No QR Code detected.")
        
        for obj in decoded_objects:
            # Get the data from the QR code
            logger.info(f"Decoded data: {obj.data}")
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
    
if __name__ == "__main__":
    # Load the image
    frame = cv2.imread("test.jpg")
    
    # Process the image for UPI ID
    upi_id = get_upi(frame)
    print(upi_id)