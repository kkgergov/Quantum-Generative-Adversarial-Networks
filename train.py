import torch
from model import LinearRegressionModel
import torch.nn as nn
import matplotlib.pyplot as plt

def generate_data():
    x = torch.linspace(0, 10, 100).unsqueeze(1)
    y = 2 * x + 1 + 0.5 * torch.randn_like(x)
    return x, y

def train_model(model, x, y, lr=0.01, epochs=200):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    return model

def plot_results(x, y, model):
    with torch.no_grad():
        pred = model(x)
    plt.scatter(x, y, label='Data')
    plt.plot(x, pred, color='red', label='Fitted Line')
    plt.legend()
    plt.show()