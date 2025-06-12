import cv2
import numpy as np
from ultralytics import YOLO
from typing import List,Dict,Any

class HelmetDetector:
    def __init__(self,model_path : str = "helmet.pt",conf_threshold : float = 0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_map = {0 : "helmet", 1:"no helmet"}

    def detect_violations(self, motorcycle : Dict) ->List[Dict]:
        """
        Detect helmet violations in a detected motorcycle
        Args:
            motorcycle: Vehicle dictionary from VehicleDetector
        Returns:
            List of violations with details
        """
        violations = []
        cropped_img = motorcycle['cropped_img']


        if cropped_img.size == 0:
            return violations

        #Detect helmets
        results = self.model(cropped_img, conf = self.conf_threshold, verbose = False)[0]
        has_helmet = False
        riders = []

        for detection in results.boxes:
            class_id = int(detection.cls)
            conf = float(detection.conf)
            bbox = detection.xyxy[0].cpu().numpy().astype(int)

            if class_id == 0:
                has_helmet = True
            elif class_id == 1:
                riders.append({
                    'roi_bbox' : bbox,
                    'confidence' : conf
                })

        #creating violations
        for rider in riders:
            violation = {
                'motorcycle_bbox': motorcycle['bbox'],
                'rider_bbox': self._map_to_original(rider['roi_bbox'], motorcycle['bbox']),
                'confidence': rider['confidence']
            }
            violations.append(violation)

        return violations
    
    def map_to_original(self, roi_bbox : np.ndarray, moto_bbox : List[int])->List[int]:
        """
            Convert ROI cords to original ones
        """
        x1_m, y1_m,x2_m,y2_m = moto_bbox
        w_m = x2_m - y2_m
        h_m = x1_m - y1_m


        return [
            x1_m + int(roi_bbox[0] * (w_m/640)),
            y1_m + int(roi_bbox[1] + (h_m/640)),
            x1_m + int(roi_bbox[2] * (w_m/640)),
            y1_m + int(roi_bbox[3] + (h_m/640)),
        ]
    
    def visualize_violations(self, frame: np.ndarray, violations: List[Dict]) -> np.ndarray:
        """
        Draw helmet violations on the frame
        """
        annotated = frame.copy()
        
        for violation in violations:
            # Draw motorcycle box
            x1, y1, x2, y2 = violation['motorcycle_bbox']
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw rider box
            rx1, ry1, rx2, ry2 = violation['rider_bbox']
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 165, 255), 2)
            
            # Add label
            label = f"No Helmet: {violation['confidence']:.2f}"
            cv2.putText(annotated, label, (rx1, ry1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return annotated