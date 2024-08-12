from PIL import Image
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import torch.nn as nn

NUM_CLASSES = 2

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Instantiate the model
model = fasterrcnn_resnet50_fpn(pretrained=False)

# Get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Replace the head with a new one, according to your number of classes
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

model.to(device)

checkpoint = torch.load("faster_rcnn_model.pth")
model.load_state_dict(checkpoint)
model.eval()

# Load an image
print("loading image...")
img = Image.open("./IMG_9695.jpg").convert("RGB")
T = transforms.ToTensor()
img = T(img).unsqueeze(0).to(device)

# Inference
print("inferencing...")
with torch.no_grad():
    prediction = model(img)

print("done.")

print("prediction = ", prediction)

# Display the bounding boxes on the image
boxes = prediction[0]['boxes'].cpu().numpy()
scores = prediction[0]['scores'].cpu().numpy()
labels = prediction[0]['labels'].cpu().numpy()


for i, box in enumerate(boxes):
    if scores[i] > 0.5:  # confidence threshold
        print(f"Detected object {labels[i]} with confidence {scores[i]}")
        print(f"Bounding box: {box}")
