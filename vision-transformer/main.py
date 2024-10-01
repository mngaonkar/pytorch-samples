import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class ViT(nn.Module):
    def __init__(self, num_classes=10, patch_size=16, in_channel=3, embed_dim=128, num_head=8, mlp_dim=256):
        super(ViT, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # patch embedding
        self.patch_embedding = nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size)
        
        # class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # transformer encoder
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_head, dim_feedforward=mlp_dim, dropout=0.1)
        
        # classification head
        self.classification_head = nn.Linear(embed_dim, num_classes)    

    def forward(self, x):
        x = self.patch_embedding(x)
        batch_size, _, height, width = x.shape
        patches = torch.flatten(x, 1, -1).view(batch_size, -1, self.embed_dim)
        class_token = self.class_token.repeat(1, patches.size(1), 1)

        # concatenate class token and patches
        patches = torch.cat([patches, class_token], dim=1)

        # transformer encoder
        patches = self.transformer_encoder(patches.transpose(0, 1)).transpose(0, 1)

        global_average_pooling = torch.mean(patches[:,1:], dim=1)

        output = self.classification_head(global_average_pooling)

        return output
    
model = ViT().to('mps')
print("model loaded in GPU")
image_path = 'in.jpg'
image = Image.open(image_path)
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
image = preprocess(image).unsqueeze(0)
print("image pre-processed")

image = image.to('mps')
try:
	output = model(image)
	print(output)
except Exception as e:
	print(e)


