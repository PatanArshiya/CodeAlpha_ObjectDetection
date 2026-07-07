import cv2
from ultralytics import YOLO

# -----------------------------
# Load YOLOv8 Model
# -----------------------------

model = YOLO("yolov8n.pt")

# -----------------------------
# Open Webcam
# -----------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot access webcam.")
    exit()

# -----------------------------
# Video Writer
# -----------------------------

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

if fps == 0:
    fps = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    "output.mp4",
    fourcc,
    fps,
    (width, height)
)

print("Press Q to quit.")

# -----------------------------
# Detection Loop
# -----------------------------

while True:

    success, frame = cap.read()

    if not success:
        break

    # Object Detection + Tracking
    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    # Draw Results
    annotated_frame = results[0].plot()

    # Save Video
    out.write(annotated_frame)

    # Display
    cv2.imshow(
        "CodeAlpha Object Detection & Tracking",
        annotated_frame
    )

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()