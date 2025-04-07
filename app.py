import cv2
from flask import Flask, Response, render_template
from ultralytics import YOLO
import datetime
import csv
import time

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov10n.pt")

csv_filename = "detection_log.csv"

# Write CSV header if file is empty
with open(csv_filename, "a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Class", "Confidence", "X1", "Y1", "X2", "Y2"])

last_logged_time = time.time()

def get_webcam():
    """Tries different webcam inputs until a successful one is found."""
    for camera_index in range(10):  # Try up to 10 camera indices
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"✅ Webcam found at index: {camera_index}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Increase resolution
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap
        else:
            cap.release()
    print("❌ ERROR: Could not find any working webcam.")
    return None

# Function to generate video frames with object detection from webcam
def generate_frames():
    global last_logged_time

    cap = get_webcam()

    if cap is None:
        yield b''
        return

    while True:
        try:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("⚠️ WARNING: Failed to capture frame from webcam.")
                continue

            # Correct the blue tint (common in webcams)
            frame_rgb_corrected = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = model(frame_rgb_corrected)
            detected_objects = []

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        label = f"{model.model.names[cls]} {conf:.2f}"

                        # Draw bounding boxes and labels on the ORIGINAL frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        detected_objects.append([
                            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                            model.model.names[cls], conf, x1, y1, x2, y2
                        ])

            if detected_objects and time.time() - last_logged_time >= 2:
                with open(csv_filename, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(detected_objects)
                last_logged_time = time.time()

            # Encode the ORIGINAL frame (with drawings) for display
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])  # Increase JPEG quality
            frame_bytes = buffer.tobytes()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

        except Exception as e:
            print(f"❌ ERROR in frame processing: {e}")
            continue
    cap.release()

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        print("Webcam feed closed.")