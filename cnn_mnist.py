import torch
import torchvision
import torchvision.transforms as transforms

# Dataset
transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

# CNN Model
model = torch.nn.Sequential(

    # Convolution Layer
    torch.nn.Conv2d(
        in_channels=1,
        out_channels=16,
        kernel_size=3,
        padding=1
    ),

    torch.nn.ReLU(),

    # Pooling
    torch.nn.MaxPool2d(kernel_size=2),

    # Flatten
    torch.nn.Flatten(),

    # Fully connected layer
    torch.nn.Linear(16 * 14 * 14, 10)
)

# Loss
loss_fn = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training
for epoch in range(3):

    for images, labels in train_loader:

        predictions = model(images)

        loss = loss_fn(predictions, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch+1}: Loss = {loss.item()}")

print("\nCNN Training finished!")