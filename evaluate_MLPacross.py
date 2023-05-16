import pandas as pd
from models import MLP_across
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import prepare_dataset_across
from sklearn.metrics import confusion_matrix

learning_rate = 0.001
batch_size = 32
epochs = 300

objects_across_train = pd.read_csv("visual_relationship/subset_data/across_images_objects_train.csv")
vrd_across_train = pd.read_csv("visual_relationship/subset_data/across_images_vrd_train.csv")
objects_across_validation = pd.read_csv("visual_relationship/subset_data/across_images_objects_validation.csv")
vrd_across_validation = pd.read_csv("visual_relationship/subset_data/across_images_vrd_validation.csv")
objects_across_test = pd.read_csv("visual_relationship/subset_data/across_images_objects_test.csv")
vrd_across_test = pd.read_csv("visual_relationship/subset_data/across_images_vrd_test.csv")

train, y_train, train_ids = prepare_dataset_across(vrd_across_train, objects_across_train)
val, y_val, val_ids = prepare_dataset_across(vrd_across_validation, objects_across_validation)
test, y_test, test_ids = prepare_dataset_across(vrd_across_test, objects_across_test)

#Keep just the distance values
y_train = y_train[:, 0]
y_val = y_val[:, 0]
y_test = y_test[:, 0]

train_dataset = torch.utils.data.TensorDataset(train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

across_model = MLP_across(train.shape[1])

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(across_model.parameters())

train_distance_f1scores = []

# Perform the training loop
for epoch in range(epochs):
    epoch_loss_avg = 0.0
    epoch_f1score_distance = 0.0

    across_model.train()

    # Iterate over the training data in batches
    for inputs, labels in train_loader:
        optimizer.zero_grad()

        dis_logits, dis_train_pred = across_model(inputs)
        loss = loss_fn(dis_logits, labels)

        loss.backward()
        optimizer.step()

        # Track progress
        epoch_loss_avg += loss.item()

        predicted_distance_labels = torch.argmax(dis_train_pred, dim=1)
        f1_distance = f1_score(labels, predicted_distance_labels, average='weighted')
        epoch_f1score_distance += f1_distance

    # Save train epoch losses and f1 scores
    epoch_loss_avg /= len(train_loader)
    epoch_f1score_distance /= len(train_loader)

    # Save validation epoch losses and f1 scores
    _, dis_val_pred= across_model(val)
    dis_val_pred = torch.argmax(dis_val_pred, dim=1)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss_avg}, F1 Score Distance: {epoch_f1score_distance}")
    print("Validation:  F1 Score Distance:", f1_score(y_val, dis_val_pred, average='weighted'))

_, dis_test_pred = across_model(test)
dis_test_pred = torch.argmax(dis_test_pred, dim=1)
print("Test Distance F1 score:", f1_score(y_test, dis_test_pred, average='weighted'), "\n")
print("Distance CM \n", confusion_matrix(y_test, dis_test_pred))
