import torch
from PIL import Image
import urllib
from torchvision import transforms

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)
input_image = Image.open(filename)
# input_image.show()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

input_batch = input_batch.to('mps')
model.to('mps')

with torch.no_grad():
    output = model(input_batch)

# print(output[0])
probabilities = torch.nn.functional.softmax(output[0], dim=0)
# print(probabilities)

with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

prob, cat = torch.topk(probabilities, 5)
for i in range(prob.size(0)):
    print(classes[cat[i]], prob[i].item())