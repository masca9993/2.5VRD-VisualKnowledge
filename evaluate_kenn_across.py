import pandas as pd
from models import Kenn_across, Kenn, MLP_across
from sklearn.metrics import f1_score
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import prepare_dataset, prepare_dataset_across2_0, transitive_triplets_across, predict_across
from sklearn.metrics import confusion_matrix


def run_across_kenn(train, y_train, train_within,
                    val, y_val, val_within,
                    test, y_test, test_within,
                    index_xy_val, index_yz_val, index_xz_val,
                    index_xy_test, index_yz_test, index_xz_test):

    learning_rate = 0.001
    batch_size = train.shape[0]
    epochs = 300
    best_f1 = 0.0
    patience = 20
    counter = 0
    kenn_layers = 5

    kenn_across_model = Kenn_across("knowledge_across", train.shape[1], kenn_layers)

    loss_fn = nn.CrossEntropyLoss()
    parameters = list(kenn_across_model.parameters())
    for i in range(kenn_layers):
        parameters += list(kenn_across_model.kenn_layers[i].parameters())
    optimizer = optim.Adam(parameters)

    train_distance_f1scores = []
    train_occlusion_f1scores = []

    kenn_within = Kenn("knowledge_within", test_within.shape[1], 3)
    kenn_within.load_state_dict(torch.load("trained_within_model"))
    kenn_within.eval()
    within_preact_train, within_pred_train, _, _ = kenn_within([train_within, torch.Tensor([]), torch.Tensor([]), torch.Tensor([])])
    within_preact_val, within_pred_val, _, _ = kenn_within([val_within, torch.Tensor([]), torch.Tensor([]), torch.Tensor([])])
    within_preact_test, within_pred_test, _, _ = kenn_within([test_within, torch.Tensor([]), torch.Tensor([]), torch.Tensor([])])

    for epoch in range(epochs):
        kenn_across_model.train()

        optimizer.zero_grad()
        dis_logits, dis_train_pred = kenn_across_model([train,
                                                        within_preact_train,
                                                        [],
                                                        [],
                                                        []]
                                                       )

        loss = loss_fn(dis_train_pred, y_train)

        loss.backward()
        optimizer.step()

        # Track progress
        epoch_loss = loss.item()

        predicted_distance_labels = torch.argmax(dis_train_pred, dim=1)
        epoch_f1score_distance = f1_score(y_train, predicted_distance_labels, average='weighted')

        kenn_across_model.eval()

        _, dis_val_pred = kenn_across_model([val, within_preact_val, index_xy_val, index_yz_val, index_xz_val])
        dis_val_pred = torch.argmax(dis_val_pred, dim=1)
        f1_val = f1_score(y_val, dis_val_pred, average='weighted')

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}, F1 Score Distance: {epoch_f1score_distance}")
        print("Validation:  F1 Score Distance:", f1_val)

        if f1_val > best_f1:
            best_f1 = f1_val
            counter = 0

        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered. Training stopped.")
                break

    kenn_across_model.eval()
    _, dis_test_pred = kenn_across_model([test, within_preact_test, index_xy_test, index_yz_test, index_xz_test])
    dis_test_pred = torch.argmax(dis_test_pred, dim=1)
    f1_test = f1_score(y_test, dis_test_pred, average='weighted')
    print("Test Distance F1 score:", f1_test, "\n")
    print("Distance CM \n", confusion_matrix(y_test, dis_test_pred))
    for name, param in kenn_across_model.kenn_layers[0].named_parameters():
        if param.requires_grad:
            print(name, param.data)

    return f1_test



'''def run_across_kenn_groundtruth(train, y_train, y_train_within,
                    val, y_val, y_val_within,
                    test, y_test, y_test_within,
                    index_xy_val, index_yz_val, index_xz_val,
                    index_xy_test, index_yz_test, index_xz_test):

    learning_rate = 0.001
    epochs = 300
    best_f1 = 0.0
    patience = 10
    counter = 0

    kenn_across_model = Kenn_across("knowledge_across_groundtruth", train.shape[1], 3)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(kenn_across_model.parameters())




    for epoch in range(epochs):
        kenn_across_model.train()

        optimizer.zero_grad()
        dis_logits, dis_train_pred = kenn_across_model([train,
                                                        y_train_withim,
                                                        [],
                                                        [],
                                                        []]
                                                       )

        loss = loss_fn(dis_train_pred, y_train)

        loss.backward()
        optimizer.step()

        # Track progress
        epoch_loss = loss.item()

        predicted_distance_labels = torch.argmax(dis_train_pred, dim=1)
        epoch_f1score_distance = f1_score(y_train, predicted_distance_labels, average='weighted')

        # Save validation epoch losses and f1 scores
        _, dis_val_pred = kenn_across_model([val, within_preact_val, index_xy_val, index_yz_val, index_xz_val])
        dis_val_pred = torch.argmax(dis_val_pred, dim=1)
        f1_val = f1_score(y_val, dis_val_pred, average='weighted')

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}, F1 Score Distance: {epoch_f1score_distance}")
        print("Validation:  F1 Score Distance:", f1_val)

        if f1_val > best_f1:
            best_f1 = f1_val
            counter = 0

        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered. Training stopped.")
                break

    _, dis_test_pred = kenn_across_model([test, within_preact_test, index_xy_test, index_yz_test, index_xz_test])
    dis_test_pred = torch.argmax(dis_test_pred, dim=1)
    f1_test = f1_score(y_test, dis_test_pred, average='weighted')
    print("Test Distance F1 score:", f1_test, "\n")
    return f1_test
'''