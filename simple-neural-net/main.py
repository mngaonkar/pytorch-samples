import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plts


class ConvNet(nn.Module):
    """ConvNet class defines the neural network model."""
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1) # input: 28x28x1, output: 26x26x16
        self.conv2 = nn.Conv2d(16, 32, 3, 1) # input: 26x26x16, output: 24x24x32
        self.dropout1 = nn.Dropout(0.10) 
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(4608, 64) # input: 24x24x32, output: 64 - maxpooling
        self.fc2 = nn.Linear(64, 10) # input: 64, output: 10
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
# same model as above but using sequential API
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, 1),
    nn.ReLU(),
    nn.Conv2d(16, 32, 3, 1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout(0.10),
    nn.Flatten(),
    nn.Linear(4608, 64),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(64, 10),
    nn.LogSoftmax(dim=1)
).to('mps')

def train(model, device, train_dataloader, optim, epoch):
    model.train()
    for index, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        optim.zero_grad()
        pred = model(X)
        loss = F.nll_loss(pred, y)
        loss.backward()
        optim.step()
        if index % 10 == 0:
            print(f"Train Epoch: {epoch} [{index * len(X)}/{len(train_dataloader.dataset)} ({100. * index / len(train_dataloader):.0f}%)]\tLoss: {loss.item():.6f}")


def test(model, device, test_dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += F.nll_loss(pred, y, reduction='sum').item()
            pred_class = pred.argmax(dim=1, keepdim=True)
            correct += pred_class.eq(y.view_as(pred_class)).sum().item()
    
    test_loss /= len(test_dataloader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_dataloader.dataset)} ({100. * correct / len(test_dataloader.dataset):.0f}%)\n")

# Load the MNIST dataset
train_dataloader = torch.utils.data.DataLoader(datasets.MNIST("../data",
                                                             train=True,
                                                             download=True,
                                                             transform=transforms.Compose([
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize((0.1307,), (0.3081,))
                                                             ])),
                                               batch_size=32, shuffle=True)

test_dataloader = torch.utils.data.DataLoader(datasets.MNIST("../data",
                                                            train=False,
                                                            transform=transforms.Compose([
                                                                transforms.ToTensor(),
                                                                transforms.Normalize((0.1307,), (0.3081,))
                                                            ])),
                                              batch_size=32, shuffle=True)
# Define optimizer
optim = optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(1, 3):
    train(model, 'mps', train_dataloader, optim, epoch)
    test(model, 'mps', test_dataloader)

test_samples, _ = next(iter(test_dataloader))
test_samples = test_samples.to('mps')
output = model(test_samples)
pred = output.argmax(dim=1)
fig, axes = plts.subplots(4, 8, figsize=(20, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(test_samples[i].cpu().numpy().reshape(28, 28), cmap='gray')
    ax.set_title(f"Predicted: {pred[i].item()}")
    ax.axis('off')

plts.show()


