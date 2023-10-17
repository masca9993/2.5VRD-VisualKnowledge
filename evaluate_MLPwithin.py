import pandas as pd
from models import MLP_within2
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import prepare_dataset
from sklearn.metrics import confusion_matrix


def run_MLPwithin(train, y_train, val, y_val, test, y_test):

    learning_rate = 0.001
    batch_size = 128
    epochs = 100
    best_f1_dis = 0.0
    best_f1_occ = 0.0
    patience = 10
    counter_dis = 0
    counter_occ = 0
    early_stop_dis = False
    early_stop_occ = False

    train_dataset = torch.utils.data.TensorDataset(train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    distance_within_model = MLP_within2(train.shape[1])
    occlusion_within_model = MLP_within2(train.shape[1])

    loss_dis = nn.CrossEntropyLoss()
    loss_occ = nn.CrossEntropyLoss()
    optimizer_distance = optim.Adam(distance_within_model.parameters())
    optimizer_occlusion = optim.Adam(occlusion_within_model.parameters())


    # Perform the training loop
    for epoch in range(epochs):
        epoch_loss_dis = 0.0
        epoch_loss_occ = 0.0
        epoch_f1score_distance = 0.0
        epoch_f1score_occlusion = 0.0

        distance_within_model.train()
        occlusion_within_model.train()

        if not early_stop_dis:

            for inputs, labels in train_loader:
                optimizer_distance.zero_grad()

                dis_logits, dis_train_pred = distance_within_model(inputs)
                loss_distance = loss_dis(dis_train_pred, labels[:, 0])

                loss_distance.backward()
                optimizer_distance.step()

                epoch_loss_dis += loss_distance.item()

                predicted_distance_labels = torch.argmax(dis_train_pred, dim=1)
                f1_distance = f1_score(labels[:, 0], predicted_distance_labels, average='weighted')
                epoch_f1score_distance += f1_distance

        if not early_stop_occ:

            for inputs, labels in train_loader:
                optimizer_occlusion.zero_grad()

                occ_logits, occ_train_pred = occlusion_within_model(inputs)
                loss_occlusion = loss_occ(occ_train_pred, labels[:, 1])

                loss_occlusion.backward()
                optimizer_occlusion.step()

                epoch_loss_occ += loss_occlusion.item()

                predicted_occlusion_labels = torch.argmax(occ_train_pred, dim=1)
                f1_occlusion = f1_score(labels[:, 1], predicted_occlusion_labels, average='weighted')
                epoch_f1score_occlusion += f1_occlusion

        if early_stop_dis and early_stop_occ:
            break


        # Save train epoch losses and f1 scores
        epoch_loss_dis /= len(train_loader)
        epoch_loss_occ /= len(train_loader)
        epoch_f1score_distance /= len(train_loader)
        epoch_f1score_occlusion /= len(train_loader)

        distance_within_model.eval()
        occlusion_within_model.eval()
        # Save validation epoch losses and f1 scores
        _, dis_val_pred = distance_within_model(val)
        _, occ_val_pred = occlusion_within_model(val)
        dis_val_pred = torch.argmax(dis_val_pred, dim=1)
        occ_val_pred = torch.argmax(occ_val_pred, dim=1)

        f1_dis_val = f1_score(y_val[:, 0], dis_val_pred, average='weighted')
        f1_occ_val = f1_score(y_val[:, 1], occ_val_pred, average='weighted')

        print(f"Epoch {epoch + 1}/{epochs}, Loss Distance: {epoch_loss_dis}, Loss Occlusion: {epoch_loss_occ}, F1 Score Distance: {epoch_f1score_distance}, F1 Score Occlusion: {epoch_f1score_occlusion}")
        print("Validation:  F1 Score Distance:", f1_dis_val, ", F1 Score Occlusion:", f1_occ_val)

        if f1_dis_val > best_f1_dis:
            best_f1_dis = f1_dis_val
            counter_dis = 0

        else:
            counter_dis += 1
            if counter_dis >= patience:
                early_stop_dis = True

        if f1_occ_val > best_f1_occ:
            best_f1_occ = f1_occ_val
            counter_occ = 0

        else:
            counter_occ += 1
            if counter_occ >= patience:
                early_stop_occ = True


    distance_within_model.eval()
    occlusion_within_model.eval()
    _, dis_test_pred = distance_within_model(test)
    _, occ_test_pred = occlusion_within_model(test)
    dis_test_pred = torch.argmax(dis_test_pred, dim=1)
    occ_test_pred = torch.argmax(occ_test_pred, dim=1)
    f1_dis = f1_score(y_test[:, 0], dis_test_pred, average='weighted')
    f1_occ = f1_score(y_test[:, 1], occ_test_pred, average='weighted')
    print("Test Distance F1 score:", f1_dis)
    print("Test Occlusion F1 score:", f1_occ, "\n")
    '''print("Distance CM \n", confusion_matrix(y_test[:, 0], dis_test_pred))
    print("Occlusion CM \n", confusion_matrix(y_test[:, 1], occ_test_pred))'''
    return f1_dis, f1_occ
