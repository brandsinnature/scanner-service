import json
import openai_func

def detect_objects(frame, client):
    products = openai_func.detect_object_openai(client, frame)
    
    result = {
        "detections": products,
    }

    return result

# def main():
#     frame_data = input()
    
#     image_data = base64.b64decode(frame_data.split(',')[1])  # Removing the "data:image/jpeg;base64," part

#     # Convert the byte data into an OpenCV image
#     image_array = np.frombuffer(image_data, dtype=np.uint8)
#     image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
#     # Simulate detection
#     detection_result = detect_objects(image)

#     # Print JSON output (for API/subprocess consumption)
#     print(json.dumps(detection_result))

# if __name__ == "__main__":
#     main()
    
# import cv2
# import torch
# from torchvision.transforms import transforms
# from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
# import numpy as np
# from PIL import Image

# class ObjectDetector:
#     def __init__(self):
#         # Load pre-trained model
#         self.model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
#         self.model.eval()
        
#         # COCO dataset class labels
#         self.CLASSES = [
#             'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#             'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign',
#             'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
#             'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack',
#             'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee',
#             'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#             'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
#             'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
#             'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
#             'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table',
#             'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote',
#             'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
#             'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors',
#             'teddy bear', 'hair drier', 'toothbrush'
#         ]
        
#         # Set up image transforms
#         self.transform = transforms.Compose([
#             transforms.ToTensor()
#         ])

#     def detect_objects(self, frame):
#         # Convert frame to RGB (OpenCV uses BGR)
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Convert to PIL Image and apply transforms
#         image = Image.fromarray(frame_rgb)
#         image_tensor = self.transform(image)
        
#         # Add batch dimension
#         image_tensor = image_tensor.unsqueeze(0)
        
#         with torch.no_grad():
#             predictions = self.model(image_tensor)
            
#         pred_boxes = predictions[0]['boxes'].detach().numpy()
#         pred_classes = predictions[0]['labels'].detach().numpy()
#         pred_scores = predictions[0]['scores'].detach().numpy()
        
#         return pred_boxes, pred_classes, pred_scores

# def main():
#     # Initialize camera
#     cap = cv2.VideoCapture(0)
#     detector = ObjectDetector()
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         # Get predictions
#         boxes, classes, scores = detector.detect_objects(frame)
        
#         # Filter predictions with confidence > 0.5
#         mask = scores > 0.5
#         boxes = boxes[mask]
#         classes = classes[mask]
#         scores = scores[mask]
        
#         # Draw predictions on frame
#         for box, cls, score in zip(boxes, classes, scores):
#             x1, y1, x2, y2 = box.astype(int)
#             label = f"{detector.CLASSES[cls]}: {score:.2f}"
            
#             # Draw bounding box
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
#             # Draw label
#             cv2.putText(frame, label, (x1, y1-10),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#         # Show frame
#         cv2.imshow('Object Detection', frame)
        
#         # Break loop with 'q' key
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     # Clean up
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()