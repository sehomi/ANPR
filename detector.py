
import cv2
from ultralytics import YOLO
import easyocr
import time

from utils import visualize_plate, license_complies_format, format_license


class PlateDetector:
    
    def __init__(self, car_model_path, plate_model_path):
        
        # Create all models 
        self.car_model = YOLO(car_model_path)
        self.plate_model = YOLO(plate_model_path)
        self.reader = easyocr.Reader(['en'], gpu=True)

        # A dictionary relating a tracked car id to a licence plate
        self.database = {}


    def detect(self, frame):

        # Make a copy for visualizations
        vis_frame = frame.copy()

        # Detect cars
        results = self.car_model.track(frame,persist=True, verbose=False)
        
        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, id, score, label = result

            # Check if threshold met and object is a car
            if score > 0.3 and label in [2, 3, 5, 7]:

                # Crop each car to search for licence plate
                cropped_img = frame[int(y1):int(y2), int(x1):int(x2)]

                # Detect plate on a car
                plates = self.plate_model(cropped_img, verbose=False)

                max_score = 0
                ocr_res = None
                for plate in plates[0].boxes.data.tolist():
                    if score > 0.6:
                        x1_plate, y1_plate, x2_plate, y2_plate, score_plate, _ = plate

                        # Draw a blue rect on around the plate
                        cv2.rectangle(vis_frame, (int(x1+x1_plate), int(y1+y1_plate)), (int(x1+x2_plate), int(y1+y2_plate)), (255, 0, 0), 6)

                        # To minimize bad readings, only read the plates when vehicle are close to camera.
                        if y1_plate + y1 < 0.66*frame.shape[0]:
                            continue

                        # Ignore the plate of the next car in the line by checking approximate location of the plate w.r.t the car.
                        if y1_plate < (y2 - y1) / 2:
                            continue

                        # Crop the plate on the car
                        lp_crop = cropped_img[int(y1_plate):int(y2_plate), int(x1_plate):int(x2_plate)]

                        # Some preprocessings on the plate image to increase correct ocr
                        lp_crop_gray = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)
                        # lp_crop_gray = cv2.equalizeHist(lp_crop_gray)
                        _, lp_crop_gray = cv2.threshold(lp_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                        
                        # Detect characters
                        ocr_results = self.reader.readtext(lp_crop_gray)

                        for detection in ocr_results:
                            bbox, text, score = detection

                            # All characters on a plate must be uppercase
                            text = text.upper().replace(' ', '')

                            # Check if the detected string complies with a normal plate. 
                            # It checks the length and placing of digits and chars.
                            # If digits and chars are mistakenly placed, some possible
                            # variations are checked (like 0 and o).
                            # For each car, the licence with maximum score is saved.
                            if license_complies_format(text):
                                formatted_text = format_license(text)
                                if score > max_score and score > 0.3:
                                    max_score = score
                                    ocr_res = formatted_text

                # In case of successful plate recognition, it will be added to a dictionary.
                # The dictionary is neccessary so that if car's plate is occluded in some frames,
                # we can preserve its licence based on its tracking id.   
                if ocr_res is not None:
                    self.database[id] = ocr_res
                
                # Visualizations.
                if id in self.database:
                    cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                    text= self.database[id]
                    visualize_plate(vis_frame, ((int(x1), int(y1)), (int(x2), int(y2))), text)
                else:
                    cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)


        return vis_frame