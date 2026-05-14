import torch

# Trainingsdaten
x = torch.tensor([[2.0], [4.0], [6.0]])
y = torch.tensor([[3.0], [6.0], [9.0]])

# Modell
model = torch.nn.Linear(1, 1)

# Fehlerfunktion
loss_fn = torch.nn.MSELoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training
for epoch in range(1000):

    prediction = model(x)

    loss = loss_fn(prediction, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")

print("Learned weight:", model.weight.item())
print("Learned bias:", model.bias.item())

test = torch.tensor([[50.0]])

print("Prediction for 50:", model(test).item())

# Tesor: 1) Wichtigste Datenstruktur in AI 2) kann Zahlen, Vektoren oder Matrizen speichern 3) Fast alles in AI basiert auf Tensors
# Trainingsdaten: Model lernt muster aus den Daten 
# Weight: Wichtigkeit, Bias: Verschiebung, Epoch: Trainingsrunde   