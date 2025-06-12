import cv2
import numpy as np
from ultralytics import YOLO
from typing import List,Dict,Any

class SeatBeltDetector:
    def __init__(self, model_path: str = "seatbelt.pt", conf_threshold : float = 0.5):
        """
            Initialize seatbelt detector
        """
        self.model=YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_map = {0:"seatbelt", 1:"no-seatbelt"}

    def detect_violations(self, car:Dict) ->List[Dict]:
        violations = []
        cropped_img = car['cropped_img']

        if cropped_img.size == 0:
            return violations
        
        result = self.model(cropped_img, self.conf_threshold, verbose = False)[0]
        has_seatbelt = False

        occupants = []

        for detection in result.boxes:
            class_id = int(detection.cls)
            conf = float(detection.conf)
            bbox = detection.xyxy[0].cpu().numpy().astype(int)

            if class_id == 0:
                has_seatbelt = True
            elif class_id == 1:
                occupants.append({
                    'roi_bbox' : bbox,
                    'confidence' : conf
                })

        for occupant in occupants:
            violation = {
                'car_bbox': car['bbox'],
                'occupant_bbox': self._map_to_original(occupant['roi_bbox'], car['bbox']),
                'confidence': occupant['confidence']
            }
            violations.append(violation)
            
        return violations
    
    def _map_to_original(self, roi_bbox: np.ndarray, car_bbox: List[int]) -> List[int]:
        """
        Convert ROI coordinates to original image coordinates
        """
        x1_c, y1_c, x2_c, y2_c = car_bbox
        w_c = x2_c - x1_c
        h_c = y2_c - y1_c
        
        return [
            x1_c + int(roi_bbox[0] * (w_c / 640)),  # x1 in original
            y1_c + int(roi_bbox[1] * (h_c / 640)),  # y1 in original
            x1_c + int(roi_bbox[2] * (w_c / 640)),  # x2 in original
            y1_c + int(roi_bbox[3] * (h_c / 640))   # y2 in original
        ]

    
    def visualize_violations(self, frame: np.ndarray, violations: List[Dict]) -> np.ndarray:
        """
        Draw seatbelt violations on the frame
        """
        annotated = frame.copy()
        
        for violation in violations:
            # Draw car bounding box
            x1, y1, x2, y2 = violation['car_bbox']
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw occupant bounding box
            rx1, ry1, rx2, ry2 = violation['occupant_bbox']
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 165, 255), 2)
            
            # Add label
            label = f"No Seatbelt: {violation['confidence']:.2f}"
            cv2.putText(annotated, label, (rx1, ry1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return annotated
    

    