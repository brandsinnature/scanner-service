import cv2
import numpy as np
import re

def extract_upi_info(image):
    """
    Extract UPI ID and name from a QR code image using OpenCV
    
    Args:
        image: OpenCV image object (already loaded with cv2.imread)
        
    Returns:
        dict: Dictionary containing 'upi_id' and 'name' if found, empty strings otherwise
    """
    result = {"upi_id": "", "name": ""}
    
    try:
        # Verify that image is not None
        if image is None:
            print("Error: Invalid image provided")
            return result
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use QRCodeDetector from cv2
        qr_detector = cv2.QRCodeDetector()
        val, points, straight_qrcode = qr_detector.detectAndDecode(gray)
        
        if val:
            print(f"QR Code detected. Raw content: {val}")
            
            # Parse UPI information
            # Common UPI format: upi://pay?pa=upi_id@provider&pn=Person%20Name&am=amount
            upi_pattern = r'pa=([^&]+)'
            name_pattern = r'pn=([^&]+)'
            
            upi_match = re.search(upi_pattern, val)
            name_match = re.search(name_pattern, val)
            
            if upi_match:
                result["upi_id"] = upi_match.group(1)
            
            if name_match:
                # Handle URL encoding in name (replace %20 with space, etc.)
                name = name_match.group(1)
                name = name.replace("%20", " ")
                result["name"] = name
                
    except Exception as e:
        print(f"Error processing QR code: {e}")
    
    return {"upi_id": result["upi_id"] if result["upi_id"] else None, "name": result["name"] if result["name"] else None}   

    
    