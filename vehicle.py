import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Any

class VehicleDetector:
    VEHICLE_CLASSES = {
        1: 'bicycle',
        2: 'car', 
        3: 'motorcycle', 
        5: 'bus', 
        7: 'truck'
    }
    
    def __init__(self, model_path="yolov8n.pt", confidence=0.5):

        self.model = YOLO(model_path)
        self.confidence = confidence
    
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict[str, Any]]:

        results = self.model(frame, conf=self.confidence)[0]
        
        # Extract detections
        vehicles = []
        
        for detection in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = detection
            
            # Filter only vehicle classes
            if int(class_id) in self.VEHICLE_CLASSES:
                vehicle_type = self.VEHICLE_CLASSES[int(class_id)]
                
                # Create bounding box coordinates (convert to integers for OpenCV)
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                
                # Create vehicle object
                vehicle = {
                    'bbox': bbox,
                    'confidence': conf,
                    'type': vehicle_type,
                    'cropped_img': frame[int(y1):int(y2), int(x1):int(x2)]
                }
                
                vehicles.append(vehicle)
        
        return vehicles
    
    def visualize(self, frame: np.ndarray, vehicles: List[Dict[str, Any]]) -> np.ndarray:

        output_frame = frame.copy()
        
        for vehicle in vehicles:
            bbox = vehicle['bbox']
            vehicle_type = vehicle['type']
            confidence = vehicle['confidence']
            
            # Draw bounding box
            cv2.rectangle(
                output_frame, 
                (bbox[0], bbox[1]), 
                (bbox[2], bbox[3]), 
                (0, 255, 0), 
                2
            )
            
            # Draw label
            label = f"{vehicle_type}: {confidence:.2f}"
            cv2.putText(
                output_frame, 
                label, 
                (bbox[0], bbox[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )
        
        return output_frame
    
    def resize_for_display(self, frame, max_width=1000, max_height=700):

        height, width = frame.shape[:2]
        
        # Calculate the ratio of the width and height
        ratio_w = max_width / width
        ratio_h = max_height / height
        
        # Use the smaller ratio to ensure the image fits within max dimensions
        ratio = min(ratio_w, ratio_h)
        
        # Only resize if the image is larger than the max dimensions
        if ratio < 1:
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            frame = cv2.resize(frame, (new_width, new_height))
            
        return frame

if __name__ == "__main__":
    detector = VehicleDetector()
    
    cap = cv2.VideoCapture(r'C:\Users\ASUS\Desktop\VS code\Web\Killers\TraffixAI\www.mp4')  # Replace with video file path if needed
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect vehicles
        vehicles = detector.detect_vehicles(frame)
        
        # Draw detections
        output_frame = detector.visualize(frame, vehicles)
        
        # Resize for display (add this line)
        display_frame = detector.resize_for_display(output_frame)
        
        # Display information
        print(f"Detected {len(vehicles)} vehicles")
        for i, vehicle in enumerate(vehicles):
            print(f"Vehicle {i+1}: {vehicle['type']}, Confidence: {vehicle['confidence']:.2f}")
        
        # Show results (use the resized frame)
        cv2.imshow("Vehicle Detection", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()