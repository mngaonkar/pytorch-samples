import json
import os
import torch

def labelme_to_fasterrcnn(labelme_json_file, output_file, class_map):
    with open(labelme_json_file, 'r') as f:
        data = json.load(f)
    
    # Initialize lists to hold converted annotations
    boxes = []
    labels = []
    
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        
        if len(points) == 2:  # It's a bounding box annotation
            xmin = min(points[0][0], points[1][0])
            ymin = min(points[0][1], points[1][1])
            xmax = max(points[0][0], points[1][0])
            ymax = max(points[0][1], points[1][1])
        else:  # If it's a polygon, convert it to a bounding box
            x_points = [p[0] for p in points]
            y_points = [p[1] for p in points]
            xmin = min(x_points)
            ymin = min(y_points)
            xmax = max(x_points)
            ymax = max(y_points)

        boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
        labels.append(class_map[label])

    with open(output_file, 'w') as file:
        for index in range(len(boxes)):
            xmin, ymin, xmax, ymax = boxes[index]
            label = labels[index]
            file.write(f"{xmin} {ymin} {xmax} {ymax} {label}\n")

# Example usage
labelme_dir = "/Users/mahadevgaonkar/code/pytorch-samples/object-detection-rcnn/dataset/annotations"
output_dir = "./dataset/labels"
class_map = {"LED": 0, "Cable": 1}  # Define your class mappings

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(labelme_dir):
    if filename.endswith('.json'):
        labelme_json_file = os.path.join(labelme_dir, filename)
        output_file = os.path.join(output_dir, filename.replace('.json', '.pt'))
        labelme_to_fasterrcnn(labelme_json_file, output_file, class_map)
