import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Transform images to tensors
transform = transforms.ToTensor()

# Load CIFAR10 dataset
train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# DataLoader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

# CIFAR10 classes
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# CNN Model
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),

    torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),

    torch.nn.Flatten(),

    torch.nn.Linear(64 * 8 * 8, 128),
    torch.nn.ReLU(),

    torch.nn.Linear(128, 10)
)

# Loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 3

for epoch in range(epochs):

    for images, labels in train_loader:

        predictions = model(images)

        loss = loss_fn(predictions, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch+1} completed | Loss: {loss.item()}")

print("\nTraining finished!")

# Evaluation mode
model.eval()

# Get one image
image, label = train_dataset[0]

# Disable gradients
with torch.no_grad():

    prediction = model(image.unsqueeze(0))

    probabilities = torch.softmax(prediction, dim=1)

    predicted_class = torch.argmax(probabilities).item()

    confidence = probabilities[0][predicted_class].item()

# Show image
plt.imshow(image.permute(1, 2, 0))

plt.title(
    f"Real: {classes[label]} | Predicted: {classes[predicted_class]}\nConfidence: {confidence*100:.2f}%"
)

plt.axis("off")

plt.show()

# Print prediction
print("\nReal class:", classes[label])

print("Predicted class:", classes[predicted_class])

print(f"Confidence: {confidence*100:.2f}%")