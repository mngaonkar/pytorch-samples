import cv2

def draw_bounding_boxes(image_path, label_path, class_list):
    # Read the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    
    # Open the label file
    with open(label_path, 'r') as file:
        lines = file.readlines()
        
        # Process each line in the label file
        for line in lines:
            # Split the line into components
            components = line.strip().split()
            class_id = int(components[0])
            x_center = float(components[1])
            y_center = float(components[2])
            bbox_width = float(components[3])
            bbox_height = float(components[4])
            
            # Convert YOLO format to image coordinates
            x_center = int(x_center * width)
            y_center = int(y_center * height)
            bbox_width = int(bbox_width * width)
            bbox_height = int(bbox_height * height)
            
            # Calculate the top-left corner of the bounding box
            x_min = int(x_center - bbox_width / 2)
            y_min = int(y_center - bbox_height / 2)
            x_max = int(x_center + bbox_width / 2)
            y_max = int(y_center + bbox_height / 2)
            
            # Draw the bounding box on the image
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=color_list[class_id], thickness=2)
            
            # Put the class label text on the bounding box
            label_text = class_list[class_id]
            cv2.putText(image, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_list[class_id], 2)
    
    # Display the image with bounding boxes
    cv2.imshow("Image with Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = '/Users/mahadevgaonkar/code/pytorch-samples/object-detection-yolo/YOLODataset/images/train/IMG_9736.jpg'
label_path = '/Users/mahadevgaonkar/code/pytorch-samples/object-detection-yolo/YOLODataset/labels/train/IMG_9736.txt'
class_list = ['LED', 'Cable']  # Replace with your classes
color_list = [(0, 255, 0), (0, 0, 255)]  # Replace with your colors

draw_bounding_boxes(image_path, label_path, class_list)
