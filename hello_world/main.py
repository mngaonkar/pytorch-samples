import torch

# check if MPS is available
if torch.backends.mps.is_available():
    print("MPS is available")
    print("MPS devices: ", torch.device("mps"))
else:
    print("MPS is not available")

