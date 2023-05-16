import pandas as pd
from models import MLP_within
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import prepare_dataset
from sklearn.metrics import confusion_matrix

learning_rate = 0.001
batch_size = 32
epochs = 70


objects_within_train = pd.read_csv("visual_relationship/subset_data/within_image_objects_train.csv")
vrd_within_train = pd.read_csv("visual_relationship/subset_data/within_image_vrd_train.csv")
objects_within_validation = pd.read_csv("visual_relationship/subset_data/within_image_objects_validation.csv")
vrd_within_validation = pd.read_csv("visual_relationship/subset_data/within_image_vrd_validation.csv")
objects_within_test = pd.read_csv("visual_relationship/subset_data/within_image_objects_test.csv")
vrd_within_test = pd.read_csv("visual_relationship/subset_data/within_image_vrd_test.csv")

train, y_train, train_ids = prepare_dataset(vrd_within_train, objects_within_train)
val, y_val, val_ids = prepare_dataset(vrd_within_validation, objects_within_validation)
test, y_test, test_ids = prepare_dataset(vrd_within_test, objects_within_test)

train_dataset = torch.utils.data.TensorDataset(train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

within_model = MLP_within(train.shape[1])

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(within_model.parameters())

train_distance_f1scores = []
train_occlusion_f1scores = []

# Perform the training loop
for epoch in range(epochs):
    epoch_loss_avg = 0.0
    epoch_f1score_distance = 0.0
    epoch_f1score_occlusion = 0.0

    within_model.train()

    # Iterate over the training data in batches
    for inputs, labels in train_loader:
        optimizer.zero_grad()

        dis_logits, dis_train_pred, occ_logits, occ_train_pred = within_model(inputs)
        loss_distance = loss_fn(dis_logits, labels[:, 0])
        loss_occlusion = loss_fn(occ_logits, labels[:, 1])
        loss = loss_distance + loss_occlusion

        loss.backward()
        optimizer.step()

        # Track progress
        epoch_loss_avg += loss.item()

        predicted_distance_labels = torch.argmax(dis_train_pred, dim=1)
        predicted_occlusion_labels = torch.argmax(occ_train_pred, dim=1)
        f1_distance = f1_score(labels[:, 0], predicted_distance_labels, average='weighted')
        f1_occlusion = f1_score(labels[:, 1], predicted_occlusion_labels, average='weighted')
        epoch_f1score_distance += f1_distance
        epoch_f1score_occlusion += f1_occlusion

    # Save train epoch losses and f1 scores
    epoch_loss_avg /= len(train_loader)
    epoch_f1score_distance /= len(train_loader)
    epoch_f1score_occlusion /= len(train_loader)

    # Save validation epoch losses and f1 scores
    _, dis_val_pred, _, occ_val_pred = within_model(val)
    dis_val_pred = torch.argmax(dis_val_pred, dim=1)
    occ_val_pred = torch.argmax(occ_val_pred, dim=1)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss_avg}, F1 Score Distance: {epoch_f1score_distance}, F1 Score Occlusion: {epoch_f1score_occlusion}")
    print("Validation:  F1 Score Distance:", f1_score(y_val[:, 0], dis_val_pred, average='weighted'), ", F1 Score Occlusion:", f1_score(y_val[:, 1], occ_val_pred, average='weighted'))



_, dis_test_pred, _, occ_test_pred = within_model(test)
dis_test_pred = torch.argmax(dis_test_pred, dim=1)
occ_test_pred = torch.argmax(occ_test_pred, dim=1)
print("Test Distance F1 score:", f1_score(y_test[:, 0], dis_test_pred, average='weighted'), "\n")
print("Test Occlusion F1 score:", f1_score(y_test[:, 1], occ_test_pred, average='weighted'), "\n")
print("Distance CM \n", confusion_matrix(y_test[:, 0], dis_test_pred))
print("Occlusion CM \n", confusion_matrix(y_test[:, 1], occ_test_pred))