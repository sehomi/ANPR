

import cv2
from detector import PlateDetector

cap = cv2.VideoCapture("video/sample.mp4")

# Creating a video writer
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video_path = 'video/result.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Creating a licence plate detector
pd = PlateDetector('model/yolov8n.pt', 'model/licence_plate.pt')

ret = True
counter = 0
while ret:
    # Read a frame from the camera
    ret, frame = cap.read()
    if ret and counter % 200 == 0:
        frame = pd.detect(frame)
        out.write(frame)

    counter += 1