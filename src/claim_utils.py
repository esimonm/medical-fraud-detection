import pickle
import time
import torch
from torch import nn
from torch.serialization import save

# Custom Pooling Layer
def pool(x, mode=None):
    if mode == "lse":
        out = torch.log(torch.mean(torch.exp(x), dim=0, keepdim=True))[0]
        return out
    elif mode == "mean":
        out = torch.mean(x, dim=0, keepdim=True)[0]
        return out
    else:
        out = torch.max(x, dim=0, keepdim=True)[0]
        return out


# Bag Loss
def bag_loss(predicted, truth):
    loss = nn.BCELoss()
    truth = truth.max().float()
    out = loss(predicted.squeeze(), truth)
    return out


# One train loop instance
def mil_train_loop(X, y, model, optimizer, DEVICE):
    # Setup
    model.train()
    optimizer.zero_grad()

    # Metrics
    mean_loss = 0
    con_mat = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    # Loop over bags:
    for ibag, bag in enumerate(X):
        # forward
        y_pred = model(torch.LongTensor(bag).to(DEVICE))
        # loss
        loss = bag_loss(y_pred, torch.tensor(y[ibag]).to(DEVICE))
        # back-propagation
        loss.backward()
        optimizer.step()

        # tracking
        mean_loss += loss.item()

        y_ibag = max(y[ibag])
        y_pred_item = round(y_pred.item())  # remember to look at this threshold as a hyperparam

        if y_pred_item == y_ibag:
            if y_ibag == 1:
                con_mat["tp"] += 1
            else:
                con_mat["tn"] += 1
        else:
            if y_ibag == 1:
                con_mat["fn"] += 1
            else:
                con_mat["fp"] += 1

    mean_loss = mean_loss/len(X)

    accuracy = (con_mat["tp"]+con_mat["tn"]) / (con_mat["tp"]+con_mat["fn"]+con_mat["tn"]+con_mat["fp"])

    TPR = con_mat["tp"] / (con_mat["tp"]+con_mat["fn"])
    FPR = con_mat["fp"] / (con_mat["fp"]+con_mat["tn"])

    return mean_loss, accuracy, TPR, FPR, con_mat


# One valid loop instance
def mil_valid_loop(X_valid, y_valid, model, DEVICE):
    # Setup
    model.eval()

    # Metrics
    mean_loss = 0
    con_mat = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    with torch.no_grad():
        # Loop over bags:
        for ibag, bag in enumerate(X_valid):
            # forward
            y_pred = model(torch.LongTensor(bag).to(DEVICE))
            # loss
            val_loss = bag_loss(y_pred, torch.tensor(y_valid[ibag]).to(DEVICE))

            # tracking
            mean_loss += val_loss.item()

            y_ibag = max(y_valid[ibag])
            y_pred_item = round(y_pred.item())

            if y_pred_item == y_ibag:
                if y_ibag == 1:
                    con_mat["tp"] += 1
                else:
                    con_mat["tn"] += 1
            else:
                if y_ibag == 1:
                    con_mat["fn"] += 1
                else:
                    con_mat["fp"] += 1

    mean_loss = mean_loss / len(X_valid)

    accuracy = (con_mat["tp"]+con_mat["tn"]) / (con_mat["tp"]+con_mat["fn"]+con_mat["tn"]+con_mat["fp"])

    TPR = con_mat["tp"] / (con_mat["tp"]+con_mat["fn"])
    FPR = con_mat["fp"] / (con_mat["fp"]+con_mat["tn"])

    return mean_loss, accuracy, TPR, FPR, con_mat


# Function to train passed model
def mil_train_model(
        model, optimizer, X, y, X_valid, y_valid, EPOCHS=40, START_EPOCH=0, DEVICE='cuda', trial=None,
        checkpoint_path=''):
    print('Initializing training...')
    torch.cuda.empty_cache()
    # Metrics
    losses, val_losses = [None for i in range(EPOCHS)], [None for i in range(EPOCHS)]
    accuracies, val_accuracies = [None for i in range(EPOCHS)], [None for i in range(EPOCHS)]
    TPRs, val_TPRs = [None for i in range(EPOCHS)], [None for i in range(EPOCHS)]
    FPRs, val_FPRs = [None for i in range(EPOCHS)], [None for i in range(EPOCHS)]
    con_mats, val_con_mats = [None for i in range(EPOCHS)], [None for i in range(EPOCHS)]
    f1_scores, val_f1_scores = [None for i in range(EPOCHS)], [None for i in range(EPOCHS)]

    for epoch in range(START_EPOCH, START_EPOCH + EPOCHS, 1):
        epoch_idx = epoch-START_EPOCH
        # TRAIN ---------------------------------------------
        train_start = time.time()

        mean_loss, accuracy, TPR, FPR, con_mat = mil_train_loop(X, y, model, optimizer, DEVICE)

        losses[epoch_idx] = mean_loss
        accuracies[epoch_idx] = accuracy
        TPRs[epoch_idx] = TPR
        FPRs[epoch_idx] = FPR
        con_mats[epoch_idx] = con_mat
        f1_scores[epoch_idx] = 2*con_mat['tp'] / (2*con_mat['tp'] + con_mat['fp'] + con_mat['fn'])

        train_time = time.time() - train_start

        # VALID ---------------------------------------------
        valid_start = time.time()

        val_mean_loss, val_accuracy, val_TPR, val_FPR, val_con_mat = mil_valid_loop(X_valid, y_valid, model, DEVICE)

        val_losses[epoch_idx] = val_mean_loss
        val_accuracies[epoch_idx] = val_accuracy
        val_TPRs[epoch_idx] = val_TPR
        val_FPRs[epoch_idx] = val_FPR
        val_con_mats[epoch_idx] = val_con_mat
        val_f1_scores[epoch_idx] = 2*val_con_mat['tp'] / (2*val_con_mat['tp'] + val_con_mat['fp'] + val_con_mat['fn'])

        valid_time = time.time() - valid_start

        epoch_time = train_time + valid_time

        print(
            f"{(epoch+1):3d}\{(START_EPOCH+EPOCHS):3d} | Time: {epoch_time:5.2f} | TRAIN: Loss: {mean_loss:.4f}  Accuracy: {accuracy:.4f}  TPR: {TPR:.4f}  FPR: {FPR:.4f} F1: {f1_scores[epoch_idx]:.4f} Time: {train_time:5.2f} | VALID: Loss: {val_mean_loss:.4f}  Accuracy: {val_accuracy:.4f}  TPR: {val_TPR:.4f}  FPR: {val_FPR:.4f} F1: {val_f1_scores[epoch_idx]:.4f} Time: {valid_time:5.2f}")

    metrics = {
        'train': {
            'losses': losses,
            'accuracies': accuracies,
            'TPRs': TPRs,
            'FPRs': FPRs,
            'con_mats': con_mats,
            'f1_scores': f1_scores
        },
        'valid': {
            'losses': val_losses,
            'accuracies': val_accuracies,
            'TPRs': val_TPRs,
            'FPRs': val_FPRs,
            'con_mats': val_con_mats,
            'f1_scores': val_f1_scores
        }
    }

    print('Training completed!')

    # Checkpoint for further training
    if checkpoint_path != '':
        save_checkpoint(model, optimizer, START_EPOCH+EPOCHS, losses[-1], checkpoint_path)

    if trial is not None:
        return val_TPR

    return model, metrics


# Function to save checkpoints while training
def save_checkpoint(model, optimizer, epochs, loss, path):
    print('Creating checkpoint...')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print('Checkpoint created!')
    return
