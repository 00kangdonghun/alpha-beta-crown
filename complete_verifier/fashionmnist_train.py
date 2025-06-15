import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from custom.fashion_model_data import SimpleMLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

transform = transforms.Compose([transforms.ToTensor()])
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./datasets', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True)

model.train()
for epoch in range(5):
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'complete_verifier/models/fashionmnist_mlp.pth')
