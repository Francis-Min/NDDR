import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score, precision_score, f1_score


def eval(y_pred, y_true, average='macro'):
    # numpy
    if torch.is_tensor(y_pred):
        y_pred_label = y_pred.argmax(dim=1).cpu().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.long().cpu().numpy()
    else:
        y_pred_label = np.argmax(y_pred, axis=1)

    y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
    y_score = np.max(y_pred, axis=1)
    std = np.std(y_score)

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred_label)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred_label)

    # ROC AUC
    roc_auc = roc_auc_score(y_true, y_pred, average=average, multi_class='ovr')

    # Sensitivity/Recall
    sensitivity = recall_score(y_true, y_pred_label, average=average)

    # Specificity
    if conf_matrix[0, 0] + conf_matrix[0, 1] == 0:
        specificity = 0
    else:
        specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

    # Precision
    pre = precision_score(y_true, y_pred_label, average=average, zero_division=1)

    # F1 Scores
    f1 = f1_score(y_true, y_pred_label, average=average)

    num_classes = len(np.unique(y_true))
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    for i in range(num_classes):
        y_true_class = (y_true == i)
        y_pred_class = (y_pred_label == i)
        precision[i] = precision_score(y_true_class, y_pred_class, zero_division=1)
        recall[i] = recall_score(y_true_class, y_pred_class, zero_division=1)

    return {
        "Accuracy": accuracy,
        "Confusion Matrix": conf_matrix,
        "ROC AUC": roc_auc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": pre,
        "F1 Score": f1,
        "STD": std,
    }, precision, recall


def eval_acc(y_pred, y_true, average='macro'):
    if torch.is_tensor(y_pred):
        y_pred_label = y_pred.argmax(dim=1).cpu().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.long().cpu().numpy()
    else:
        y_pred_label = np.argmax(y_pred, axis=1)

    y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)

    y_score = np.max(y_pred, axis=1)
    std = np.std(y_score)

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred_label)

    return accuracy
