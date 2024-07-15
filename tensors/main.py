import torch

# create a tensor on MPS device
points = torch.tensor([[1.0, 4.0], [2.0, 1.0], [3.0, 5.0]]).to('mps:0')

# print the tensor
print(points[0])
print("shape = ", points.shape)
print("size = ", points.size())

print(points.storage())