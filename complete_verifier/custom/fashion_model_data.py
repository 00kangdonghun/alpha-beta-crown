import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import os

# MLP 모델 정의 (Pooling/Conv 없이)
class SimpleMLP(nn.Module):
    def __init__(self, in_features=784, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, out_dim)
    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def fashionmlp(in_features=784, out_dim=10):
    return SimpleMLP(in_features, out_dim)

# FashionMNIST 데이터로더 (기존 코드 참고)
def fashionmnist(spec, use_bounds=False):
    import arguments
    eps = spec["epsilon"]
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../datasets')
    mean = torch.tensor(arguments.Config["data"]["mean"])
    std = torch.tensor(arguments.Config["data"]["std"])
    normalize = transforms.Normalize(mean=mean, std=std)
    test_data = datasets.FashionMNIST(database_path, train=False, download=True,
                                      transform=transforms.Compose([transforms.ToTensor(), normalize]))
    testloader = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=False)
    X, labels = next(iter(testloader))
    if use_bounds:
        absolute_max = ((1. - mean) / std).reshape(1, -1, 1, 1)
        absolute_min = ((0. - mean) / std).reshape(1, -1, 1, 1)
        new_eps = (eps / std).reshape(1, -1, 1, 1)
        data_max = torch.min(X + new_eps, absolute_max)
        data_min = torch.max(X - new_eps, absolute_min)
        ret_eps = None
    else:
        data_max = ((1. - mean) / std).reshape(1, -1, 1, 1)
        data_min = ((0. - mean) / std).reshape(1, -1, 1, 1)
        ret_eps = (eps / std).reshape(1, -1, 1, 1)
    return X, labels, data_max, data_min, ret_eps
