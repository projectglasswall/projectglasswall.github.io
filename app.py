from djitellopy import Tello
import cv2
from flask import Flask, Response, render_template
from ultralytics import YOLO
import datetime
import csv
import time
import threading

app = Flask(__name__)

# Initialize Tello
tello = Tello()

# Connect to the drone and start streaming
try:
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")
    tello.streamon()
except Exception as e:
    print(f"Error connecting to Tello: {e}")
    tello = None

# Load YOLO model
model = YOLO("yolov10n.pt")

csv_filename = "detection_log.csv"

# Write CSV header if file is empty
with open(csv_filename, "a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Class", "Confidence", "X1", "Y1", "X2", "Y2"])

last_logged_time = time.time()
flight_started = False
flight_start_time = 0
flight_thread = None
stop_flight = False

# Function to execute the flight path in a separate thread
def execute_flight_path():
    global tello, flight_started, flight_start_time, stop_flight
    try:
        print("Takeoff!")
        tello.takeoff()
        time.sleep(1)

        distance = 20  # Distance to move in cm
        speed = 20    # Drone speed in cm/s

        print("Move left")
        if stop_flight: return
        tello.go_xyz_speed(-distance, 0, 0, speed)
        time.sleep(1)

        print("Move right")
        if stop_flight: return
        tello.go_xyz_speed(distance, 0, 0, speed)
        time.sleep(1)

        print("Move left")
        if stop_flight: return
        tello.go_xyz_speed(-distance, 0, 0, speed)
        time.sleep(1)

        print("Move right")
        if stop_flight: return
        tello.go_xyz_speed(distance, 0, 0, speed)
        time.sleep(1)

        flight_start_time = time.time()
        flight_started = True

        time.sleep(5)  # Land after 5 seconds
        if stop_flight: return
        tello.land()
        print("Landing successful.")

        flight_started = False
        stop_flight = False

    except Exception as flight_error:
        print(f"Flight failed: {flight_error}")

# Function to generate video frames with object detection
def generate_frames():
    global last_logged_time, tello, flight_thread, stop_flight

    if tello is None:
        print("❌ ERROR: Tello is not initialized. Video feed cannot start.")
        yield b''
        return

    if not flight_started and (flight_thread is None or not flight_thread.is_alive()):
        flight_thread = threading.Thread(target=execute_flight_path)
        flight_thread.start()

    while True:
        try:
            frame = tello.get_frame_read().frame

            if frame is None:
                print("⚠️ WARNING: Failed to capture frame.")
                continue

            results = model(frame)
            detected_objects = []

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        label = f"{model.model.names[cls]} {conf:.2f}"

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

            battery = tello.get_battery() if tello else "N/A"
            cv2.putText(frame, f"Battery: {battery}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode(".jpg", frame_rgb)
            frame_bytes = buffer.tobytes()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

        except Exception as e:
            print(f"❌ ERROR in frame processing: {e}")
            continue

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
        stop_flight = True
        if tello:
            tello.land()
            tello.streamoff()
            tello.end()
            print("Tello connection closed.")