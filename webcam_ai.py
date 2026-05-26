import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

import cv2
import matplotlib.pyplot as plt

# Load pretrained ResNet18
model = models.resnet18(pretrained=True)

# Evaluation mode
model.eval()

# ImageNet labels
from torchvision.models import ResNet18_Weights

labels = ResNet18_Weights.DEFAULT.meta["categories"]

# Transform image
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Open webcam
cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Transform image
    input_tensor = transform(rgb_frame).unsqueeze(0)

    # Disable gradients
    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        predicted_class = torch.argmax(probabilities).item()

        confidence = probabilities[predicted_class].item()

    # Prediction text
    text = f"{labels[predicted_class]} ({confidence*100:.1f}%)"

    # Put text on frame
    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show webcam
    cv2.imshow("AI Webcam Detector", frame)

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()