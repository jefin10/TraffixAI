import cv2
from models.vehicle_detection.vehicle import VehicleDetector
from models.seatbelt_detection.seatbelt import SeatBeltDetector
from models.helmet_detection.helmet import HelmetDetector

vehicle_detector=VehicleDetector()
helmet_detector = HelmetDetector()
seatbelt_detector = SeatBeltDetector()

cap = cv2.VideoCapture("ved.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    vehicles = vehicle_detector.detect_vehicles(frame)
    helmet_violations = []
    seatbelt_violations = []
    
    for vehicle in vehicles:
        #helmet detection in motorcycles
        if vehicle['type']  == 'motorcycle':
            bike_violations = helmet_detector.detect_violations(vehicle)
            helmet_violations.extend(bike_violations)

        #seatbelt detection in cars
        if vehicle['type'] == 'car':
            car_violation = seatbelt_detector.detect_violations(vehicle)
            seatbelt_violations.extend(car_violation)
    
    #Visualize
    output_frame = vehicle_detector.visualize(frame, vehicles)
    output_frame = helmet_detector.visualize_violations(output_frame, helmet_violations)
    output_frame = seatbelt_detector.visualize_violations(output_frame, seatbelt_violations)


    cv2.imshow("Helmet Detection",output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()