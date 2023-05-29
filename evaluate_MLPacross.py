import pandas as pd
from models import MLP_across
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import prepare_dataset_across, img_obj_dict, predict_across
from sklearn.metrics import confusion_matrix



def run_acrossMLP(train, y_train,
                    val, y_val,
                    test, y_test):


    learning_rate = 0.001
    batch_size = 32
    epochs = 300
    best_f1 = 0.0
    patience = 20
    counter = 0

    across_model = MLP_across(train.shape[1])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(across_model.parameters())

    train_distance_f1scores = []

    # Perform the training loop
    for epoch in range(epochs):
        epoch_loss_avg = 0.0
        epoch_f1score_distance = 0.0

        across_model.train()

        optimizer.zero_grad()
        dis_logits, dis_train_pred = across_model(train)
        loss = loss_fn(dis_train_pred, y_train)

        loss.backward()
        optimizer.step()

        # Track progress
        epoch_loss = loss.item()

        predicted_distance_labels = torch.argmax(dis_train_pred, dim=1)

        epoch_f1score = f1_score(y_train, predicted_distance_labels, average='weighted')


        # Save validation epoch losses and f1 scores
        _, dis_val_pred = across_model(val)
        dis_val_pred = torch.argmax(dis_val_pred, dim=1)
        f1_val = f1_score(y_val, dis_val_pred, average='weighted')

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}, F1 Score Distance: {epoch_f1score}")
        print("Validation:  F1 Score Distance:", f1_val)

        if f1_val > best_f1:
            best_f1 = f1_val
            counter = 0

        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered. Training stopped.")
                break

    _, dis_test_pred = across_model(test)
    dis_test_pred = torch.argmax(dis_test_pred, dim=1)
    f1_test = f1_score(y_test, dis_test_pred, average='weighted')
    print("Test Distance F1 score:", f1_test, "\n")
    #print("Distance CM \n", confusion_matrix(y_test, dis_test_pred))

    return f1_test
