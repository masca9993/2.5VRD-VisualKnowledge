import cv2
import numpy as np

# Load the pre-trained model for object detection
net = cv2.dnn.readNetFromDarknet('C:/Users/UTENTE/Dropbox/PC/Downloads/yolov3.cfg', 'C:/Users/UTENTE/Dropbox/PC/Downloads/yolov3.weights')

# Load the COCO class labels
with open('C:/Users/UTENTE/Dropbox/PC/Desktop/coco.names.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load the image you want to detect objects in
image = cv2.imread('C:/Users/UTENTE/Dropbox/PC/Desktop/Georgia5and120loop.jpg')

'''
# Set the input size of the model
input_size = (1664, 1664)

# Create a blob from the image
blob = cv2.dnn.blobFromImage(image, 1/255.0, input_size, swapRB=True, crop=False)

# Set the input blob for the model
net.setInput(blob)

# Get the output layers of the model
layer_names = net.getLayerNames()
print(layer_names)
print(net.getUnconnectedOutLayers())
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

outs = net.forward(output_layers)

# Get the bounding boxes, confidences, and class IDs
boxes = []
confidences = []
class_ids = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.2:
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            width = int(detection[2] * image.shape[1])
            height = int(detection[3] * image.shape[0])
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            boxes.append([left, top, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression to remove duplicate bounding boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Loop over the indices and draw bounding boxes around the detected objects
for i in indices:
    i = i
    box = boxes[i]
    left = box[0]
    top = box[1]
    width = box[2]
    height = box[3]
    cv2.rectangle(image, (left, top), (left+width, top+height), (0, 255, 0), 2)

    # Add the label to the bounding box
    label = "{}: {:.2f}%".format(classes[class_ids[i]], confidences[i] * 100)
    cv2.putText(image, label, (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the image with bounding boxes
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# Set the input size of the model
input_size = (1664, 1664)

# Create a blob from the image
blob = cv2.dnn.blobFromImage(image, 1/255.0, input_size, swapRB=True, crop=False)

# Set the input blob for the model
net.setInput(blob)

# Get the output layers of the model
layer_names = net.getLayerNames()
print(layer_names)
print(net.getUnconnectedOutLayers())
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

outs = net.forward(output_layers)

# Get the bounding boxes, confidences, and class IDs
boxes = []
confidences = []
class_ids = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            width = int(detection[2] * image.shape[1])
            height = int(detection[3] * image.shape[0])
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            boxes.append([left, top, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Find the index of the box with the smallest area
if len(boxes) > 0:
    areas = [box[2]*box[3] for box in boxes]
    closest_box_index = np.argmin(areas)
    closest_box = boxes[closest_box_index]
    closest_confidence = confidences[closest_box_index]
    closest_class_id = class_ids[closest_box_index]
    
    # Draw bounding box and label for the closest object
    left, top, width, height = closest_box
    cv2.rectangle(image, (left, top), (left+width, top+height), (0, 255, 0), 2)
    label = "{}: {:.2f}%".format(classes[closest_class_id], closest_confidence * 100)
    cv2.putText(image, label, (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the image with bounding box
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# Get the coordinates of the POV
pov_x, pov_y = (-100, -100)  # Replace with the actual coordinates of the POV

# Compute the distances between the POV and the centers of the bounding boxes
distances = []
for box, score, class_id in zip(boxes, scores, class_ids):
    center_x = box[0] + box[2] / 2
    center_y = box[1] + box[3] / 2
    distance = ((center_x - pov_x) ** 2 + (center_y - pov_y) ** 2) ** 0.5
    distances.append((distance, box, score, class_id))

# Sort the distances in ascending order
distances.sort()

# Select the closest object
closest_distance, closest_box, closest_score, closest_id = distances[-1]

# Draw the bounding box and label for the closest object
left, top, width, height = closest_box
cv2.rectangle(image, (left, top), (left + width, top + height), (0, 255, 0), 2)
label = "{}: {:.2f}%".format(classes[closest_id], closest_score * 100)
cv2.putText(image, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the image with the closest object
cv2.imshow("Closest Object", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''