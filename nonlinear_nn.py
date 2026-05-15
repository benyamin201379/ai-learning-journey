import torch 

# Trainingsdaten

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[1.0], [4.0], [9.0], [16.0]])

# Neural Network 
model = torch.nn.Sequential(
    torch.nn.Linear(1,8), 
    torch.nn.ReLU(), 
    torch.nn.Linear(8,1)
)

# Loss Function 

loss_fn = torch.nn.MSELoss()

#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 

# Training
for epoch in range(5000): 
    
    prediction = model(x)
    
    loss = loss_fn(prediction, y)
    
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")
    
# Test 

test = torch.tensor([[5.0]])

print("Prediction for 5:", model(test).item())

