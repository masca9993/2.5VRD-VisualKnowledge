import os
import sys
import glob
import argparse
import matplotlib
import numpy as np
import cv2
import pandas as pd
import torchvision
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
import skimage.measure


sys.path.insert(0,"../")
sys.path.insert(1,"./")
# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import load_images
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from PyTorch.model import PTModel

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='../nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='../examples/*.jpg', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)
names = [weight.name for layer in model.layers for weight in layer.weights]
weights = model.get_weights()

keras_name = []
for name, weight in zip(names, weights):
  keras_name.append(name)

pytorch_model = PTModel().float()

# load parameter from keras
keras_state_dict = {} 
j = 0
for name, param in pytorch_model.named_parameters():
  
  if 'classifier' in name:
    keras_state_dict[name]=param
    continue

  if 'conv' in name and 'weight' in name:
    keras_state_dict[name]=torch.from_numpy(np.transpose(weights[j],(3, 2, 0, 1)))
    # print(name,keras_name[j])
    j = j+1
    continue
  
  if 'conv' in name and 'bias' in name:
    keras_state_dict[name]=torch.from_numpy(weights[j])
    # print(param.shape,weights[j].size)
    j = j+1
    continue

  if 'norm' in name and 'weight' in name:
    keras_state_dict[name]=torch.from_numpy(weights[j])
    # print(param.shape,weights[j].shape)
    j = j+1
    continue

  if 'norm' in name and 'bias' in name:
    keras_state_dict[name]=torch.from_numpy(weights[j])
    # print(param.shape,weights[j].size)
    j = j+1
    keras_state_dict[name.replace("bias", "running_mean")]=torch.from_numpy(weights[j])
    # print(param.shape,weights[j].size)
    j = j+1
    keras_state_dict[name.replace("bias", "running_var")]=torch.from_numpy(weights[j])
    # print(param.shape,weights[j].size)
    j = j+1
    continue


pytorch_model.load_state_dict(keras_state_dict)
pytorch_model.eval()
pytorch_model.cuda()

def my_DepthNorm(x, maxDepth):
    return maxDepth / x

def my_predict(model, images, minDepth=10, maxDepth=1000):

  with torch.no_grad():
    # Compute predictions
    predictions = model(images)

    # Put in expected range
  return np.clip(my_DepthNorm(predictions.cpu().numpy(), maxDepth=maxDepth), minDepth, maxDepth) / maxDepth



image_folder = '../../visual_relationship/images/images_validation_resized'
batch_size = 32
image_files = glob.glob(args.input)
#image_files = [image_files[i:i+32] for i in range(0, len(image_files), 32)]
# # Input images
output_folder = '../../visual_relationship/images/images_validation_depth'
for file in image_files:
  inputs = load_images( glob.glob(file) ).astype('float32')
  pytorch_input = torch.from_numpy(inputs[0,:,:,:]).permute(2,0,1).unsqueeze(0).cuda()
  print(inputs.shape)
  print(pytorch_input.shape)

  # print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

  # # Compute results
  output = my_predict(pytorch_model,pytorch_input[0,:,:,:].unsqueeze(0))
  plt.imshow(output[0,0,:,:])
  plt.axis('off')
  plt.savefig(os.path.join(output_folder, file[-20:]), bbox_inches='tight', pad_inches=0)
  torch.cuda.empty_cache()


'''objects_df = pd.read_csv("../../visual_relationship/subset_data/combined_file.csv")
objects_df['depth_img']=None
for index, row in objects_df.iterrows():
    image_path = os.path.join("../../visual_relationship/images/images_validation_depth/", row['image_id'] + ".jpg")
    depth_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)/255
    height, width = depth_img.shape
    x1 = int(width * row["xmin"])
    x2 = int(width * row["xmax"])
    y1 = int(height * row["ymin"])
    y2 = int(height * row["ymax"])
    objects_df.at[index, "depth_img"] = [depth_img[y1:y2, x1:x2]]'''


'''bounding_boxes = [[0.1221629977,0.3456149995,0.0881889984,0.3411940038],
                  [0.1670230031,0.4818870127,0.3920210004,0.7737870216],
                  [0.3693139851,0.5783770084,0.1232040003,0.4100930095],
                  [0.5047399998,0.8543069959,0.4417180121,0.8551099896],
                  [0.6022880077,0.8358150125,0.4553590119,0.6510930061],
                  [0.6841779947,0.9381009936,0.1762890071,0.484638989],
                  [0.1821980029,0.3951379955,0.4210200012,0.5824139714],
                  ]
height, width, _ = image_with_boxes.shape

# Draw each bounding box on the image
for box in bounding_boxes:
  x1, x2, y1, y2 = box
  x1 = int(width*x1)
  x2 = int(width*x2)
  y1 = int(height*y1)
  y2 = int(height*y2)

  cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)


#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(3, 3, figsize=(10, 10))

height, width = output[0,0,:,:].shape

objects_depth = []
for box in bounding_boxes:
  x1, x2, y1, y2 = box
  x1 = int(width*x1)
  x2 = int(width*x2)
  y1 = int(height*y1)
  y2 = int(height*y2)
  objects_depth.append(output[0,0,:,:][y1:y2, x1:x2])
  #cv2.rectangle(output[0, 0, :, :], (x1, y1), (x2, y2), (0, 255, 0), 2)

for i, o in enumerate(objects_depth):

  o = skimage.measure.block_reduce(o, (2,2), np.mean)
  o = skimage.measure.block_reduce(o, (2, 2), np.mean)
  print(o.shape)
  print(o)
  axarr[i//3, i%3].imshow(o)

print(output[0,0,:,:].shape)
print(image_with_boxes.shape)

figure2 = plt.figure(figsize=(10, 10))
plt.imshow(output[0,0,:,:])

cv2.imshow("Image with Bounding Boxes", image_with_boxes)
plt.show()

'''