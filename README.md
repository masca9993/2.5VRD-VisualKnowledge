# 2.5VRD-VisualKnowledge

#### A Neural symbolic approach to 2.5 Visual Relationship Detection

Project aims to perform an end to end visual relationship detection.
We start by performing object detection on raw images of [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html).
Then we apply a neural symbolic approach that injects prior common knowledge into a MLP trained on detected bounding boxes.

The goal is to merge the approaches shown in the [2.5 VRD reference paper](https://arxiv.org/pdf/2104.12727.pdf) to improve the results.


---
Run _utils.py_ in **preproccesing** folder to download the images from OID, remember to specify the percentage
if you want to create a subset of the training set.

Run function _create_data_subset.py_ to create a folder containing a subset
of images according to the train images that are in images/images_train.