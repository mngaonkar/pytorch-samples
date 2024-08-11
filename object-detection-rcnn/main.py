import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

model = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

image_path = 'dog.jpg'
image = Image.open(image_path)

image_tensor = preprocess(image).unsqueeze(0)
print("image preprocess shape = ", image_tensor.shape)

image_tensor = image_tensor.to('mps')
model.to('mps')

with torch.no_grad():
    output = model(image_tensor)

output = output.cpu()

fig, ax = plt.subplots(1)
ax.imshow(image)

for box in output[0]['boxes']:
    x, y, w, h = box
    rect = patches.Rectangle((x, y), w-x, h-y, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.show()