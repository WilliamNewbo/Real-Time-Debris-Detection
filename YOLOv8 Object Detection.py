# Imports
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
import time
from supervision.draw.color import ColorPalette
from supervision.detection.core import Detections, BoxAnnotator

# ---------------------------------------------------------------------------------------------------------------------
# Setup
# GPU Setup
print(tf.config.list_physical_devices())
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Create and define Window
cv2.namedWindow('window', cv2.WINDOW_NORMAL)
cv2.resizeWindow('window', 1960, 1000)

# Video Setup
#video_path = 'Videos/Multi-Event Demonstration.mp4'
video_path = 'Videos/Final Demonstration (Raw - Trimmed) - Made with Clipchamp.mp4'
#video_path = 'Videos/ALO, Emilia Romagna 2021 (2 Trimmed) - Made with Clipchamp.mp4'
#video_path = 'Videos/BOT, Emilia Romagna 2021 (2, 2 Trimmed) - Made with Clipchamp.mp4'
vid = cv2.VideoCapture(video_path)
# Get default video FPS
fps = vid.get(cv2.CAP_PROP_FPS)
# Get total number of video frames
total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)

# Initialise number of frames to 0
num_frames = 0

# Tool used to annotate objects
box_annotator = BoxAnnotator(color=ColorPalette.from_hex(['#e6194b', '#ffe119', '#66cccc']), thickness=4, text_thickness=1, text_scale=1)

# YOLOS Model Settings
model = YOLO("Models/YOLOv8/YOLOv8m Original.pt")
model.fuse()
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
# Dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names

# ---------------------------------------------------------------------------------------------------------------------
# Run

# Array to store predict times
prediction_times = []
# Timer to note start time of runtime
start_time = time.time()

while True:
    # Timer to note start time of frame
    frame_start_time = time.time()

    ret, frame = vid.read()
    if ret:
        # Timer to note start time of prediction
        predict_start = time.time()

        results = model(frame, iou=IOU_THRESHOLD, conf=CONFIDENCE_THRESHOLD)

        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        )

        # Format custom labels
        labels = [
            f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]

        # Annotate the frame with the models detections
        frame = box_annotator.annotate(frame, detections=detections, labels=labels)

        # Timer to note end time of prediction
        predict_end = time.time()
        # Calculate prediction time
        predict_time = predict_end-predict_start
        prediction_times.append(predict_time)
        print('Predict time: ' + str(predict_time))

        # Timer to note start time of frame
        frame_end_time = time.time()
        # Calculate FFS of the video
        elapsed_frame_time = frame_end_time - frame_start_time
        frame_fps = np.round(1 / elapsed_frame_time, 1)

        # Display FFS of video in top left corner
        cv2.putText(frame, f'FPS: {float(frame_fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        # Show results
        cv2.imshow("window", frame)
    else:
        break

    # Increment frame count
    num_frames += 1
    # If 'Q' is pressed, close windows
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Close all windows
cv2.destroyAllWindows()

# Timer to note start time of runtime
end_time = time.time()
# Calculate the duration of the runtime
elapsed_time = end_time - start_time
print(elapsed_time)

# Print the actual FFS of the original video and the FFS of the results video
print("Input Video FFS: " + str(fps))
print("Result Video FFS: " + str(num_frames/elapsed_time))

print("\nAverage Prediction TIme: " + str(sum(prediction_times)/len(prediction_times)))
