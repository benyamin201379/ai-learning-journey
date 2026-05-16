import torch

# Training data
hours = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0]
])

scores = torch.tensor([
    [50.0],
    [60.0],
    [70.0],
    [80.0],
    [90.0]
])

# Neural network
model = torch.nn.Sequential(
    torch.nn.Linear(1, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 1)
)

# Loss function
loss_fn = torch.nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(3000):

    prediction = model(hours)

    loss = loss_fn(prediction, scores)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 300 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")

# Test prediction
test = torch.tensor([[6.0]])

prediction = model(test)

print("\nStudy Hours:", test.item())
print("Predicted Exam Score:", prediction.item())