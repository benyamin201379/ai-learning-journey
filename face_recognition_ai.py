import cv2
import face_recognition

# Load known face image
known_image = face_recognition.load_image_file("known_face.jpg")

# Create face encoding for known image
known_encodings = face_recognition.face_encodings(known_image)

if len(known_encodings) == 0:
    print("No face found in known_face.jpg")
    exit()

known_encoding = known_encodings[0]

# Name of known person
known_name = "Benyamin"

# Open webcam
cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:

    success, frame = cap.read()

    if not success:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find faces in webcam frame
    face_locations = face_recognition.face_locations(rgb_frame)

    # Encode detected faces
    face_encodings = face_recognition.face_encodings(
        rgb_frame,
        face_locations
    )

    for face_encoding, face_location in zip(face_encodings, face_locations):

        # Compare detected face with known face
        matches = face_recognition.compare_faces(
            [known_encoding],
            face_encoding,
            tolerance=0.6
        )

        name = "Unknown"

        if matches[0]:
            name = known_name

        top, right, bottom, left = face_location

        # Draw rectangle around face
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

        # Draw name label
        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Recognition AI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
