import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Load YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = torch.hub.load('/Users/mahadevgaonkar/.cache/torch/hub/ultralytics_yolov5_master/', 'custom', source="local", path='/Users/mahadevgaonkar/.cache/torch/hub/ultralytics_yolov5_master/runs/train/exp16/weights/last.pt')

# Load an image
img = Image.open('/Users/mahadevgaonkar/Downloads/IMG_9757.jpg')

# Perform inference
results = model(img)

# Print results
results.print()  # Print results to console
results.show()   # Display results

# Save the image with bounding boxes
results.save('output.jpg')
