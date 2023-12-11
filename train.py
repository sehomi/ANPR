 
import utils
from ultralytics import YOLO

utils.convert_dataset("/content/ANPR/dataset/car-plate-detection/")

model = YOLO('yolov8n.yaml')
result = model.train(data="/content/ANPR/dataset/car-plate-detection/config.yaml",device="0",epochs=100,verbose=True,plots=True,save=True)