import torch
import torch.nn as nn

embeddings = nn.Embedding(num_embeddings=10, embedding_dim=3) # total tokens, vector size
input = torch.Tensor([1, 2, 3, 4, 5]).long()

output = embeddings(input)
print(output)