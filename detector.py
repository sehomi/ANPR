
import cv2
from ultralytics import YOLO
import easyocr
import time

from utils import visualize_plate


class PlateDetector:
    
    def __init__(self, car_model_path, plate_model_path):

        self.car_model = YOLO(car_model_path)
        self.plate_model = YOLO(plate_model_path)
        self.reader = easyocr.Reader(['en'], gpu=False)

        self.database = {}


    def detect(self, frame):

        results = self.car_model.track(frame,persist=True, verbose=True)

        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, id, score, label = result

            # Check if threshold met and object is a car
            if score > 0.5 and label==2:
                
                car_data = {}
                car_data['last_seen'] = time.time()
                car_data['box'] = (x1, y1, x2, y2)

                plate_existing = False
                if id in self.database:
                    if 'licence_plate' in self.database[id]:
                        plate_existing = True

                if not plate_existing:
                    cropped_img = frame[int(y1):int(y2), int(x1):int(x2)]

                    plates = self.plate_model(cropped_img)
                    for plate in plates[0].boxes.data.tolist():
                        if score > 0.6:
                            x1_plate, y1_plate, x2_plate, y2_plate, score_plate, _ = plate
                            lp_crop = cropped_img[int(y1_plate):int(y2_plate), int(x1_plate):int(x2_plate)]
                            lp_crop_gray = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)
                            ocr_results = self.reader.readtext(lp_crop_gray)

                            max_score = 0
                            ocr_res = None
                            for res in ocr_results:
                                bbox, text, score = res
                                if score > max_score and score > 0.5:
                                    max_score = score
                                    ocr_res = res

                            if ocr_res is not None:
                                car_data['licence_plate'] = ocr_res
                    
                self.database[id] = car_data

                visualize_plate(frame, self.database[id])

        return frame