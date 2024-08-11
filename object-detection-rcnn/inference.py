from PIL import Image
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms

device = torch.device('mps') if torch.mps.is_available() else torch.device('cpu')

# Instantiate the model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.load_state_dict(torch.load("faster_rcnn_model.pth"))
model.eval()

# Load an image
img = Image.open("/Users/mahadevgaonkar/Documents/IMG_9695.jpg").convert("RGB")
T = transforms.ToTensor()
img = T(img).unsqueeze(0).to(device)
img = T.ToTensor()(img).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    prediction = model(img)

# Display the bounding boxes on the image
boxes = prediction[0]['boxes'].cpu().numpy()
scores = prediction[0]['scores'].cpu().numpy()
labels = prediction[0]['labels'].cpu().numpy()

for i, box in enumerate(boxes):
    if scores[i] > 0.5:  # confidence threshold
        print(f"Detected object {labels[i]} with confidence {scores[i]}")
        print(f"Bounding box: {box}")
