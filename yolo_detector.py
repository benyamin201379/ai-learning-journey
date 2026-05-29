from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Webcam
cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:

    success, frame = cap.read()

    if not success:
        break

    # Run detection
    results = model(frame)

    # Draw detections
    annotated_frame = results[0].plot()

    # Show frame
    cv2.imshow(
        "YOLO Object Detector",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows() 