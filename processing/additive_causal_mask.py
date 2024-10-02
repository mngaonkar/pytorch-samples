import torch

def create_additive_causal_mask(size):
    # Create a upper triangular matrix with ones above diagonal
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    print(mask)

if __name__ == "__main__":
    create_additive_causal_mask(5)