import pandas as pd
from models import Kenn2, MLP_within2
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import prepare_dataset
from sklearn.metrics import confusion_matrix


def calculate_intersection(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[2], box2[2])
    x2 = min(box1[1], box2[1])
    y2 = min(box1[3], box2[3])
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    intersection_area = width * height
    return intersection_area


def run_within_kenn_material(train, y_train, val, y_val, test, y_test):

    learning_rate = 0.001
    batch_size = 250
    epochs = 100
    best_f1 = 0.0
    patience = 10
    counter = 0
    early_stop = False
    kenn_layers = 3
    delta = 0.02


    intersection_list = []
    for box1, box2 in zip(train[:, :4], train[:, 4:]):
        intersection_area = calculate_intersection(box1, box2)

        if intersection_area > delta:
            intersection_list.append(5)
        else:
            intersection_list.append(-5)
    intersection_train = torch.Tensor(intersection_list)
    train = torch.cat((train, torch.unsqueeze(intersection_train, dim=1)), 1)

    intersection_list = []
    for box1, box2 in zip(val[:, :4], val[:, 4:]):
        intersection_area = calculate_intersection(box1, box2)

        if intersection_area > delta:
            intersection_list.append(5)
        else:
            intersection_list.append(-5)
    intersection_val = torch.Tensor(intersection_list)

    intersection_list = []
    for box1, box2 in zip(test[:, :4], test[:, 4:]):
        intersection_area = calculate_intersection(box1, box2)

        if intersection_area > delta:
            intersection_list.append(5)
        else:
            intersection_list.append(-5)
    intersection_test = torch.Tensor(intersection_list)
    train_dataset = torch.utils.data.TensorDataset(train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    kenn_within_model = Kenn2("knowledge_material", train.shape[1],
                              MLP_within2(train.shape[1]-1), MLP_within2(train.shape[1]-1), kenn_layers)

    loss_fn = nn.CrossEntropyLoss()
    parameters = list(kenn_within_model.parameters())
    for i in range(kenn_layers):
        parameters += list(kenn_within_model.kenn_layers[i].parameters())

    optimizer = optim.Adam(parameters)

    #optimizer = optim.Adam(kenn_within_model.parameters())

    train_distance_f1scores = []
    train_occlusion_f1scores = []

    # Perform the training loop
    for epoch in range(epochs):
        epoch_loss_avg = 0.0
        epoch_f1score_distance = 0.0
        epoch_f1score_occlusion = 0.0

        kenn_within_model.train()

        if not early_stop:
            for inputs, labels in train_loader:
                optimizer.zero_grad()

                intersection_train = inputs[:, 8]

                dis_logits, dis_train_pred, occ_logits, occ_train_pred = kenn_within_model([inputs[:, :8], intersection_train])
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
        else:
            break

        # Save train epoch losses and accuracies
        epoch_loss_avg /= len(train_loader)
        epoch_f1score_distance /= len(train_loader)
        epoch_f1score_occlusion /= len(train_loader)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss_avg}, F1 Score Distance: {epoch_f1score_distance}, F1 Score Occlusion: {epoch_f1score_occlusion}")

        kenn_within_model.eval()
        _, dis_val_pred, _, occ_val_pred = kenn_within_model([val, intersection_val])
        dis_val_pred = torch.argmax(dis_val_pred, dim=1)
        occ_val_pred = torch.argmax(occ_val_pred, dim=1)
        f1_dis = f1_score(y_val[:, 0], dis_val_pred, average='weighted')
        f1_occ = f1_score(y_val[:, 1], occ_val_pred, average='weighted')
        f1_val = (f1_dis + f1_occ) / 2
        print("Validation:  F1 Score Distance:", f1_score(y_val[:, 0], dis_val_pred, average='weighted'), ", F1 Score Occlusion:", f1_score(y_val[:, 1], occ_val_pred, average='weighted'))

        if f1_val > best_f1:
            best_f1 = f1_val
            counter = 0

        else:
            counter += 1
            if counter >= patience:
                early_stop = True

    kenn_within_model.eval()
    _, dis_test_pred, _, occ_test_pred = kenn_within_model([test, intersection_test])
    dis_test_pred = torch.argmax(dis_test_pred, dim=1)
    occ_test_pred = torch.argmax(occ_test_pred, dim=1)
    f1_dis = f1_score(y_test[:, 0], dis_test_pred, average='weighted')
    f1_occ = f1_score(y_test[:, 1], occ_test_pred, average='weighted')
    print("Test Distance F1 score:", f1_dis)
    print("Test Occlusion F1 score:", f1_occ, "\n")
    print("KENN Confusion matrix")
    print("Distance CM \n", confusion_matrix(y_test[:, 0], dis_test_pred))
    print("Occlusion CM \n", confusion_matrix(y_test[:, 1], occ_test_pred))

    for name, param in kenn_within_model.kenn_layers[0].named_parameters():
        if param.requires_grad:
            print(name, param.data)
    #torch.save(kenn_within_model.state_dict(), "trained_within_model")

    return f1_dis, f1_occ
