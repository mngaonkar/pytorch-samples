import torch
import torch.nn as nn

class NeuralCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, layers=[64,32,16]):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        input_size = embedding_dim * 2

        self.mlp = nn.Sequential()
        self.mlp.append(nn.Linear(input_size, layers[0]))
        self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(layers[0], layers[1]))
        self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(layers[1], layers[2]))
        self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(layers[2], 1))

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        input = torch.cat((user_emb, item_emb), dim=-1)

        output = self.mlp(input)
        return output.squeeze()