import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt

# Load pretrained ResNet18 model
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Set model to evaluation mode
model.eval()

# Load ImageNet labels
labels = weights.meta["categories"]

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load image
image = Image.open("test_image.png")

# Prepare image
input_tensor = transform(image).unsqueeze(0)

# Disable gradient calculation
with torch.no_grad():

    # Run prediction
    outputs = model(input_tensor)

    # Convert outputs to probabilities
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Get predicted class
    predicted_class = torch.argmax(probabilities).item()

    # Confidence score
    confidence = probabilities[predicted_class].item()

# Show image
plt.imshow(image)
plt.axis("off")

plt.title(
    f"Prediction: {labels[predicted_class]}\n"
    f"Confidence: {confidence*100:.2f}%"
)

plt.show()

# Print prediction
print("\nPrediction:", labels[predicted_class])
print(f"Confidence: {confidence*100:.2f}%")
