import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def compute_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    cm = confusion_matrix(y_true, y_pred)
    return precision, recall, f1, cm
