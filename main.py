
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
output_video_path = 'video/result.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_video_path, fourcc, 20, (1280, 720))

# Creating a licence plate detector
pd = PlateDetector('model/yolov8n.pt', 'model/licence_plate.pt')

ret = True
counter = 0
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
while ret:
    # Read a frame from the camera
    ret, frame = cap.read()
    if ret and counter%3 == 0:
        frame = pd.detect(frame)

        # Show image
        frame = cv2.resize(frame, (1280, 720))
        out.write(frame)
        # cv2.imshow("preview", frame)
        # cv2.waitKey(10)

        print(f"Progress: {counter}/{length}")

    counter += 1