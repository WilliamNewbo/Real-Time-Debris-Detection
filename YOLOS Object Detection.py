# Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from transformers import YolosForObjectDetection, YolosImageProcessor, AutoModelForObjectDetection
import time
import supervision as sv
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
video_path = 'Videos/Final Demonstration (Raw - Trimmed) - Made with Clipchamp.mp4'
vid = cv2.VideoCapture(video_path)
# Get default video FPS
fps = vid.get(cv2.CAP_PROP_FPS)
# Get total number of video frames
total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)

# Initialise number of frames to 0
num_frames = 0

# Tool used to annotate objects
box_annotator = BoxAnnotator(color=ColorPalette.from_hex(['#66cccc', '#e6194b', '#ffe119']), thickness=2, text_thickness=1, text_scale=0.4)

# YOLOS Model Settings
CHECKPOINT = 'Models/YOLOS-Tiny-Original'
CONFIDENCE_THRESHOLD = 0.9
IOU_THRESHOLD = 0.2
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
model = YolosForObjectDetection.from_pretrained(CHECKPOINT)
model.to(device)

# ---------------------------------------------------------------------------------------------------------------------
# Run
# Array to store predict times
prediction_times = []
# Timer to note start time of runtime
start_time = time.time()

while True:
    # Timer to note start time of frame
    frame_start_time = time.time()

    # Read frame
    ret, frame = vid.read()
    if ret:
        # Timer to note start time of prediction
        predict_start = time.time()
        frame = cv2.resize(frame, (512, 512))
        with torch.no_grad():
            # Load image and predict
            inputs = image_processor(images=frame, return_tensors='pt').to(device)
            outputs = model(**inputs)

            # Post-process
            target_sizes = torch.tensor([frame.shape[:2]]).to(device)
            results = image_processor.post_process_object_detection(
                outputs=outputs,
                threshold=CONFIDENCE_THRESHOLD,
                target_sizes=target_sizes
            )[0]

        # Annotate
        detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=IOU_THRESHOLD)

        labels = [
            f"{model.config.id2label[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]

        # Annotate the frame with the models detections
        frame = box_annotator.annotate(frame, detections=detections, labels=labels)
        frame = cv2.resize(frame, (1200, 800))
        # Timer to note end time of prediction
        predict_end = time.time()
        predict_time = predict_end - predict_start
        # Calculate prediction time
        predict_time = predict_end - predict_start
        print('Predict time: ' + str(predict_time))
        prediction_times.append(predict_time)


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

    # If 'Q' is pressed, close window
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Close all windows
cv2.destroyAllWindows()

# Timer to note end time of runtime
end_time = time.time()
# Calculate the duration of the runtime
elapsed_time = end_time - start_time
print(elapsed_time)

# Print the actual FFS of the original video and the FFS of the results video
print("Input Video FFS: " + str(fps))
print("Result Video FFS: " + str(num_frames/elapsed_time))

print("\nAverage Prediction TIme: " + str(sum(prediction_times)/len(prediction_times)))

