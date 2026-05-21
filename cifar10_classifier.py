import torch
import torchvision
import torchvision.transforms as transforms

# Transform images into tensors
transform = transforms.ToTensor()

# Load CIFAR-10 training dataset
train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Create DataLoader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

# CIFAR-10 classes
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

# CNN model
model = torch.nn.Sequential(
    # Input: 3 x 32 x 32
    torch.nn.Conv2d(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        padding=1
    ),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),

    # After pooling: 16 x 16 x 16
    torch.nn.Conv2d(
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        padding=1
    ),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),

    # After pooling: 32 x 8 x 8
    torch.nn.Flatten(),

    torch.nn.Linear(32 * 8 * 8, 128),
    torch.nn.ReLU(),

    torch.nn.Linear(128, 10)
)

# Loss function for classification
loss_fn = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training
for epoch in range(5):

    total_loss = 0

    for images, labels in train_loader:

        predictions = model(images)

        loss = loss_fn(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")

print("\nCIFAR-10 CNN training finished!")

# Test with one training image
image, label = train_dataset[0]

with torch.no_grad():
    prediction = model(image.unsqueeze(0))
    predicted_class = torch.argmax(prediction).item()

print("\nReal class:", classes[label])
print("Predicted class:", classes[predicted_class])