import pandas as pd
from models import CombinedModel, ImageEmbeddingCNN, MLP_within
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import matplotlib
import cv2
import numpy as np
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import prepare_dataset
from sklearn.metrics import confusion_matrix
import argparse
from scipy.stats import ttest_ind
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('-nruns', help='number of runs of the test')
args = parser.parse_args()

combined_file_train = pd.read_csv("visual_relationship/subset_data/combined_file_test.csv")
combined_file_train = combined_file_train.iloc[:, 0:7]
vrd_within_train = pd.read_csv("visual_relationship/subset_data/within_image_vrd_validation.csv")
vrd_within_train = vrd_within_train[vrd_within_train['image_id_1'].isin(combined_file_train["image_id"].values)]
combined_file_test = pd.read_csv("visual_relationship/subset_data/combined_file.csv")
combined_file_test = combined_file_test.iloc[:, 0:7]
vrd_within_test = pd.read_csv("visual_relationship/subset_data/within_image_vrd_validation.csv")
vrd_within_test = vrd_within_test[vrd_within_test['image_id_1'].isin(combined_file_test["image_id"].values)]


objects_depth_train = pd.DataFrame(columns=["key", "depth_img"])
objects_depth_test = pd.DataFrame(columns=["key", "depth_img"])

for index, row in combined_file_train.iterrows():
    image_path = os.path.join("visual_relationship/images/images_validation_depth/", row['image_id'] + ".jpg")
    depth_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)/255
    height, width = depth_img.shape
    mid_height, mid_width = height // 2, width // 2
    top_left = depth_img[0:mid_height, 0:mid_width]
    top_right = depth_img[0:mid_height, mid_width:width]
    bottom_left = depth_img[mid_height:height, 0:mid_width]
    bottom_right = depth_img[mid_height:height, mid_width:width]

    combined_file_train.at[index, "depth_top_left"] = np.mean(top_left)
    combined_file_train.at[index, "depth_top_right"] = np.mean(top_right)
    combined_file_train.at[index, "depth_bottom_left"] = np.mean(bottom_left)
    combined_file_train.at[index, "depth_bottom_right"] = np.mean(bottom_right)

for index, row in combined_file_test.iterrows():
    image_path = os.path.join("visual_relationship/images/images_validation_depth/", row['image_id'] + ".jpg")
    depth_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)/255
    height, width = depth_img.shape
    mid_height, mid_width = height // 2, width // 2
    top_left = depth_img[0:mid_height, 0:mid_width]
    top_right = depth_img[0:mid_height, mid_width:width]
    bottom_left = depth_img[mid_height:height, 0:mid_width]
    bottom_right = depth_img[mid_height:height, mid_width:width]

    combined_file_test.at[index, "depth_top_left"] = np.mean(top_left)
    combined_file_test.at[index, "depth_top_right"] = np.mean(top_right)
    combined_file_test.at[index, "depth_bottom_left"] = np.mean(bottom_left)
    combined_file_test.at[index, "depth_bottom_right"] = np.mean(bottom_right)

train_within, y_train_within, train_ids_within = prepare_dataset(vrd_within_train, combined_file_train)
test_within, y_test_within, test_ids_within = prepare_dataset(vrd_within_test, combined_file_test)

combined_file_train = pd.read_csv("visual_relationship/subset_data/combined_file_test.csv")
combined_file_train = combined_file_train.iloc[:, 0:7]
vrd_within_train = pd.read_csv("visual_relationship/subset_data/within_image_vrd_validation.csv")
vrd_within_train = vrd_within_train[vrd_within_train['image_id_1'].isin(combined_file_train["image_id"].values)]
combined_file_test = pd.read_csv("visual_relationship/subset_data/combined_file.csv")
combined_file_test = combined_file_test.iloc[:, 0:7]
vrd_within_test = pd.read_csv("visual_relationship/subset_data/within_image_vrd_validation.csv")
vrd_within_test = vrd_within_test[vrd_within_test['image_id_1'].isin(combined_file_test["image_id"].values)]

train_bb, y_train_bb, train_ids_bb = prepare_dataset(vrd_within_train, combined_file_train)
test_bb, y_test_bb, test_ids_bb = prepare_dataset(vrd_within_test, combined_file_test)

def run_MLPwithin(train, y_train, val, y_val, test, y_test):

    learning_rate = 0.01
    batch_size = 64
    epochs = 100
    best_f1 = 0.0

    patience = 60
    counter_dis = 0
    counter_occ = 0
    early_stop = False


    mlp_model = MLP_within(train.shape[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_model.to(device)
    train = train.to(device)
    y_train = y_train.to(device)

    train_dataset = torch.utils.data.TensorDataset(train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp_model.parameters())

    for epoch in range(epochs):
        epoch_loss_avg = 0.0
        epoch_f1score_distance = 0.0
        epoch_f1score_occlusion = 0.0

        mlp_model.train()

        if not early_stop:
            for inputs, labels in train_loader:

                optimizer.zero_grad()

                dis_logits, dis_train_pred, occ_logits, occ_train_pred = mlp_model(inputs)
                loss_distance = loss_fn(dis_logits, labels[:, 0])
                loss_occlusion = loss_fn(occ_logits, labels[:, 1])
                loss = loss_distance + loss_occlusion

                loss.backward()
                optimizer.step()

                # Track progress
                epoch_loss_avg += loss.item()

                predicted_distance_labels = torch.argmax(dis_train_pred, dim=1).cpu()
                predicted_occlusion_labels = torch.argmax(occ_train_pred, dim=1).cpu()


                f1_distance = f1_score(labels[:, 0].cpu(), predicted_distance_labels, average='weighted')
                f1_occlusion = f1_score(labels[:, 1].cpu(), predicted_occlusion_labels, average='weighted')
                epoch_f1score_distance += f1_distance
                epoch_f1score_occlusion += f1_occlusion
        else:
            break

        # Save train epoch losses and f1 scores
        epoch_loss_avg /= len(train_loader)
        epoch_f1score_distance /= len(train_loader)
        epoch_f1score_occlusion /= len(train_loader)

        mlp_model.eval()
        _, dis_val_pred, _, occ_val_pred = mlp_model(train)
        dis_val_pred = torch.argmax(dis_val_pred, dim=1).cpu()
        occ_val_pred = torch.argmax(occ_val_pred, dim=1).cpu()
        f1_dis = f1_score(y_train[:, 0].cpu(), dis_val_pred, average='weighted')
        f1_occ = f1_score(y_train[:, 1].cpu(), occ_val_pred, average='weighted')
        f1_val = (f1_dis + f1_occ) / 2
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss_avg}, F1 Score Distance: {epoch_f1score_distance}, F1 Score Occlusion: {epoch_f1score_occlusion}")
        print("Validation:  F1 Score Distance:", f1_dis, ", F1 Score Occlusion:", f1_occ)

        if f1_val > best_f1:
            best_f1 = f1_val
            counter = 0

        else:
            counter += 1
            if counter >= patience:
                early_stop = True

    test = test.to(device)

    mlp_model.eval()
    _, dis_test_pred, _, occ_test_pred = mlp_model(test)
    dis_test_pred = torch.argmax(dis_test_pred, dim=1).cpu()
    occ_test_pred = torch.argmax(occ_test_pred, dim=1).cpu()


    f1_dis_test = f1_score(y_test[:, 0].cpu(), dis_test_pred, average='weighted')
    f1_occ_test = f1_score(y_test[:, 1].cpu(), occ_test_pred, average='weighted')

    print("Test Distance F1 score:", f1_dis_test, "\n")
    print("Test Occlusion F1 score:", f1_occ_test, "\n")
    print("MLP Confusion matrix")
    print("Distance CM \n", confusion_matrix(y_test[:, 0], dis_test_pred))
    print("Occlusion CM \n", confusion_matrix(y_test[:, 1], occ_test_pred))

    #torch.save(combined_model.state_dict(), "trained_within_model")
    return f1_dis_test, f1_occ_test



n_runs = args.nruns
f1_distance_depth = []
f1_occlusion_depth = []

f1_distance_sift = []
f1_occlusion_sift = []

for i in range(int(n_runs)):
    print("\n -Starting run: ", i)
    sift_dis, sift_occ = run_MLPwithin(train_bb, y_train_bb, None, None, test_bb, y_test_bb)
    depth_dis, depth_occ = run_MLPwithin(train_within, y_train_within, None, None, test_within, y_test_within)

    f1_distance_depth.append(depth_dis)
    f1_occlusion_depth.append(depth_occ)
    f1_distance_sift.append(sift_dis)
    f1_occlusion_sift.append(sift_occ)


print("Depth average f1 score Distance:", sum(f1_distance_depth)/len(f1_distance_depth))
print("BB average f1 score Distance", sum(f1_distance_sift)/len(f1_distance_sift), "\n")
print("Depth average f1 score Occlusion:", sum(f1_occlusion_depth)/len(f1_occlusion_depth))
print("BB average f1 score Occlusion", sum(f1_occlusion_sift)/len(f1_occlusion_sift))

_, pvalue_dis = ttest_ind(f1_distance_depth, f1_distance_sift, equal_var=True)
_, pvalue_occ = ttest_ind(f1_occlusion_depth, f1_occlusion_sift, equal_var=True)
print("p value Distance: ", pvalue_dis)
print("p value Occlusion: ", pvalue_occ)

