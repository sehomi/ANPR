
import os
import cv2
from detector import PlateDetector
import time

video_path = "video/sample.mp4"
if os.path.isfile(video_path):
    cap = cv2.VideoCapture(video_path)
else:
    print("Error: video does not exist.")
    exit()

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
    if ret and counter%2 == 0:
        frame = pd.detect(frame)
        out.write(frame)

        # Show image
        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("preview", frame)
        cv2.waitKey(10)

    counter += 1