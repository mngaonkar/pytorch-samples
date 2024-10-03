import torch
import torch.nn as nn

class GatedFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GatedFeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.activation(out1)
        gate = self.sigmoid(self.gate(x))
        out3 = gate * out2
        out = self.fc2(out3)

        return out
    
if __name__ == "__main__":
    model = GatedFeedForward(5, 10, 1)
    input_tensor = torch.rand(3, 5)
    output = model(input_tensor)
    print(output)


