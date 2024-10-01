import torch
import torch.nn as nn

logits = torch.tensor([[0.1, 0.2, 0.3, 0], [0.2, 0.3, 0.4, 0], [0.3, 0.4, 0.5, 0], [0.3, 0.4, 0.5, 0]])
labels = torch.tensor([0, 1, 2, 3])

loss_fn = nn.CrossEntropyLoss()

try:
    loss = loss_fn(logits, labels)
    print(loss)
except Exception as e:
    print(e)