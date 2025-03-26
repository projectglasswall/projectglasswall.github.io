from flask import Flask, Response, render_template
import cv2
import torch
import datetime
import csv
import time
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov10n.pt")

# Open webcam
cap = cv2.VideoCapture(1)

csv_filename = "detection_log.csv"

# Open CSV file for logging
with open(csv_filename, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Class", "Confidence", "X1", "Y1", "X2", "Y2"])  # Write header only once

    last_logged_time = time.time()

# Function to generate video frames
def generate_frames():
    global last_logged_time

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)  # Run YOLO detection

        detected_objects = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = f"{model.model.names[cls]} {conf:.2f}"

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    detected_objects.append([
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],  
                        model.model.names[cls], conf, x1, y1, x2, y2
                    ])

        if detected_objects and time.time() - last_logged_time >= 2:
            with open(csv_filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(detected_objects)
            last_logged_time = time.time()

        # Encode frame as JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
