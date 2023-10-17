import pandas as pd
from models import CombinedModel, ImageEmbeddingCNN, MLP_SiftDepth
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

combined_file_train = pd.read_csv("visual_relationship/subset_data/combined_file.csv")
vrd_within_train = pd.read_csv("visual_relationship/subset_data/within_image_vrd_validation.csv")
vrd_within_train = vrd_within_train[vrd_within_train['image_id_1'].isin(combined_file_train["image_id"].values)]
combined_file_test = pd.read_csv("visual_relationship/subset_data/combined_file_test.csv")
vrd_within_test = pd.read_csv("visual_relationship/subset_data/within_image_vrd_validation.csv")
vrd_within_test = vrd_within_test[vrd_within_test['image_id_1'].isin(combined_file_test["image_id"].values)]


objects_depth_train = pd.DataFrame(columns=["key", "depth_img"])
objects_depth_test = pd.DataFrame(columns=["key", "depth_img"])


for index, row in combined_file_train.iterrows():
    image_path = os.path.join("visual_relationship/images/images_validation_depth/", row['image_id'] + ".jpg")
    depth_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)/255
    height, width = depth_img.shape
    x1 = int(width * row["xmin"])
    x2 = int(width * row["xmax"])
    y1 = int(height * row["ymin"])
    y2 = int(height * row["ymax"])
    bb_image = np.zeros_like(depth_img)
    bb_image[y1:y2, x1:x2] = depth_img[y1:y2, x1:x2]

    objects_depth_train = objects_depth_train.append({'key': row['image_id'] + "_" + str(row['object_id']), 'depth_img': bb_image}, ignore_index=True)

for index, row in combined_file_test.iterrows():
    image_path = os.path.join("visual_relationship/images/images_validation_depth/", row['image_id'] + ".jpg")
    depth_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)/255
    height, width = depth_img.shape
    x1 = int(width * row["xmin"])
    x2 = int(width * row["xmax"])
    y1 = int(height * row["ymin"])
    y2 = int(height * row["ymax"])
    bb_image = np.zeros_like(depth_img)
    bb_image[y1:y2, x1:x2] = depth_img[y1:y2, x1:x2]

    objects_depth_test = objects_depth_test.append({'key': row['image_id'] + "_" + str(row['object_id']), 'depth_img': bb_image}, ignore_index=True)

'''images = objects_depth["depth_img"].to_numpy()
objects_depth.drop('depth_img', axis=1, inplace=True)
'''
train_within, y_train_within, train_ids_within = prepare_dataset(vrd_within_train, combined_file_train)
test_within, y_test_within, test_ids_within = prepare_dataset(vrd_within_test, combined_file_test)


def run_MLPwithin(train, y_train, val, y_val, test, y_test):

    learning_rate = 0.001
    batch_size = 1
    epochs = 20
    best_f1_dis = 0.0
    best_f1_occ = 0.0
    patience = 10
    counter_dis = 0
    counter_occ = 0
    early_stop = False

    #train_dataset = torch.utils.data.TensorDataset(train, y_train)
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Create an instance of the CNN for image embeddings
    cnn_model = ImageEmbeddingCNN()

    input_size = cnn_model.fc1.out_features*2 + 208  # 100 VW + 8 coordinate di BB
    hidden_size = 1024
    output_size = 8
    mlp_model = MLP_SiftDepth(input_size, hidden_size, output_size)

    # Create an instance of the combined model
    combined_model = CombinedModel(cnn_model, mlp_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    combined_model.to(device)
    train = train.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(combined_model.parameters())

    for epoch in range(epochs):
        epoch_loss_avg = 0.0
        epoch_f1score_distance = 0.0
        epoch_f1score_occlusion = 0.0

        combined_model.train()

        if not early_stop:
            for mlp_input, object_ids, labels in zip(train, train_ids_within, y_train):

                image_depth1 = torch.tensor(objects_depth_train[objects_depth_train["key"] == object_ids[0] + "_" + str(object_ids[1])]["depth_img"].values[0], dtype=torch.float32).to(device)
                image_depth2 = torch.tensor(objects_depth_train[objects_depth_train["key"] == object_ids[2] + "_" + str(object_ids[3])]["depth_img"].values[0], dtype=torch.float32).to(device)

                image_depth1 = image_depth1.unsqueeze(0).unsqueeze(0)
                image_depth2 = image_depth2.unsqueeze(0).unsqueeze(0)
                mlp_input = mlp_input.unsqueeze(0)
                labels = labels.to(device)
                labels = labels.unsqueeze(0)

                optimizer.zero_grad()

                dis_logits, dis_train_pred, occ_logits, occ_train_pred = combined_model([image_depth1, image_depth2], mlp_input)
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
        epoch_loss_avg /= len(train_ids_within)
        epoch_f1score_distance /= len(train_ids_within)
        epoch_f1score_occlusion /= len(train_ids_within)

        dis_predictions = []
        occ_predictions = []

        for mlp_input, object_ids, labels in zip(train, train_ids_within, y_train):

            image_depth1 = torch.tensor(
                objects_depth_train[objects_depth_train["key"] == object_ids[0] + "_" + str(object_ids[1])]["depth_img"].values[0],
                dtype=torch.float32).to(device)
            image_depth2 = torch.tensor(
                objects_depth_train[objects_depth_train["key"] == object_ids[2] + "_" + str(object_ids[3])]["depth_img"].values[0],
                dtype=torch.float32).to(device)

            image_depth1 = image_depth1.unsqueeze(0).unsqueeze(0)
            image_depth2 = image_depth2.unsqueeze(0).unsqueeze(0)
            mlp_input = mlp_input.unsqueeze(0)

            combined_model.eval()

            _, dis_val_pred, _, occ_val_pred = combined_model([image_depth1, image_depth2], mlp_input)
            dis_val_pred = torch.argmax(dis_val_pred, dim=1).item()
            occ_val_pred = torch.argmax(occ_val_pred, dim=1).item()

            dis_predictions.append(dis_val_pred)
            occ_predictions.append(occ_val_pred)

        f1_dis = f1_score(y_train[:, 0].cpu(), dis_predictions, average='weighted')
        f1_occ = f1_score(y_train[:, 1].cpu(), occ_predictions, average='weighted')

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss_avg}, F1 Score Distance: {epoch_f1score_distance}, F1 Score Occlusion: {epoch_f1score_occlusion}")
        print("Validation:  F1 Score Distance:", f1_dis, ", F1 Score Occlusion:", f1_occ)

        '''        if f1_val > best_f1:
            best_f1 = f1_val
            counter = 0

        else:
            counter += 1
            if counter >= patience:
                early_stop = True'''

    dis_predictions_test = []
    occ_predictions_test = []
    test = test.to(device)

    for mlp_input, object_ids, labels in zip(test, test_ids_within, y_test):
        image_depth1 = torch.tensor(
            objects_depth_test[objects_depth_test["key"] == object_ids[0] + "_" + str(object_ids[1])][
                "depth_img"].values[0],
            dtype=torch.float32).to(device)
        image_depth2 = torch.tensor(
            objects_depth_test[objects_depth_test["key"] == object_ids[2] + "_" + str(object_ids[3])][
                "depth_img"].values[0],
            dtype=torch.float32).to(device)

        image_depth1 = image_depth1.unsqueeze(0).unsqueeze(0)
        image_depth2 = image_depth2.unsqueeze(0).unsqueeze(0)
        mlp_input = mlp_input.unsqueeze(0)

        combined_model.eval()
        _, dis_val_pred, _, occ_val_pred = combined_model([image_depth1, image_depth2], mlp_input)
        dis_val_pred = torch.argmax(dis_val_pred, dim=1).item()
        occ_val_pred = torch.argmax(occ_val_pred, dim=1).item()

        dis_predictions_test.append(dis_val_pred)
        occ_predictions_test.append(occ_val_pred)

    f1_dis_test = f1_score(y_test[:, 0].cpu(), dis_predictions_test, average='weighted')
    f1_occ_test = f1_score(y_test[:, 1].cpu(), occ_predictions_test, average='weighted')

    print("Test Distance F1 score:", f1_dis_test, "\n")
    print("Test Occlusion F1 score:", f1_occ_test, "\n")
    print("MLP Confusion matrix")
    print("Distance CM \n", confusion_matrix(y_test[:, 0], dis_predictions_test))
    print("Occlusion CM \n", confusion_matrix(y_test[:, 1], occ_predictions_test))

    #torch.save(combined_model.state_dict(), "trained_within_model")
    return f1_dis_test, f1_occ_test

for i in range(1):
    run_MLPwithin(train_within, y_train_within, None, None, test_within, y_test_within)
