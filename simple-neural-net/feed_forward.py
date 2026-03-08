import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

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
    # MPS profiling - use CPU activity and move tensors to MPS device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True)
    with profiler:
        input_tensor = torch.randn(3, 5).to(device)
        model = FeedForward(5, 10, 1).to(device)
        print(model)
        
        output = model(input_tensor)

        print(output)
    
    print(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
        