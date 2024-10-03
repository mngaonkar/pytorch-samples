import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x    
    
if __name__ == "__main__":
    input_tensor = torch.randn(3, 5)
    model = FeedForward(5, 10, 1)
    print(model)
    
    output = model(input_tensor)

    print(output)
        