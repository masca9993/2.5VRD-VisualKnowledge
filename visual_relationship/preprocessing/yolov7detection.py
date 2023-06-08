import yolov7 #pip install yolov7detct
import os 
os.chdir('C:/Users/UTENTE/Dropbox/PC/Documents/GitHub/2.5VRD-VisualKnowledge/visual_relationship/preprocessing/')

print (os.getcwd() )

# load pretrained or custom model
model = yolov7.load('kadirnar/yolov7-v0.1', hf_model=True)

# set model parameters
model.conf = 0.5  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.classes = None  # (optional list) filter by class

# set image
imgs = '../images/images_train/0a0f9a953a8b810a.jpg' 
# perform inference
results = model(imgs)

# inference with larger input size and test time augmentation
results = model(imgs, size=1280, augment=True)

# parse results
predictions = results.pred[0]
boxes = predictions[:, :4] # x1, y1, x2, y2
scores = predictions[:, 4]
categories = predictions[:, 5]

# show detection bounding boxes on image
results.show()