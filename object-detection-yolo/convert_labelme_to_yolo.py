import json
import os
import cv2

def convert_labelme_to_yolo(labelme_json_path, output_dir, class_list):
    # Load the Labelme JSON file
    with open(labelme_json_path, 'r') as f:
        data = json.load(f)
    
    # Get image dimensions
    img_height = data['imageHeight']
    img_width = data['imageWidth']
    
    # Create the YOLO annotation file
    output_txt_path = os.path.join(output_dir, os.path.splitext(os.path.basename(labelme_json_path))[0] + '.txt')
    with open(output_txt_path, 'w') as yolo_file:
        for shape in data['shapes']:
            label = shape['label']
            if label not in class_list:
                continue
            
            class_id = class_list.index(label)
            points = shape['points']
            
            # Convert polygon to bounding box
            x_points = [p[0] for p in points]
            y_points = [p[1] for p in points]
            
            x_min = min(x_points)
            x_max = max(x_points)
            y_min = min(y_points)
            y_max = max(y_points)
            
            # Calculate YOLO format coordinates
            x_center = (x_min + x_max) / 2.0 / img_width
            y_center = (y_min + y_max) / 2.0 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            
            # Write to file
            yolo_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Example usage
labelme_json_dir = '/Users/mahadevgaonkar/code/pytorch-samples/object-detection-yolo/mouse_airpod/'
output_dir = '/Users/mahadevgaonkar/code/pytorch-samples/object-detection-yolo/mouse_airpod/'
class_list = ['Airpod', 'Mouse']  # Replace with your classes

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Convert all Labelme JSON files in the directory
for filename in os.listdir(labelme_json_dir):
    if filename.endswith('.json'):
        labelme_json_path = os.path.join(labelme_json_dir, filename)
        convert_labelme_to_yolo(labelme_json_path, output_dir, class_list)
