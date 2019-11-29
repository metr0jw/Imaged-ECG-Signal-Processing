import os
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np

from PIL import Image

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import roc_curve, auc, hinge_loss, log_loss


def load_image(infilename):
    img = Image.open(infilename).convert('L')
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def plot_acc(accuracy):
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.plot([1, 2, 3, 4, 5], accuracy)
    plt.title("5-Fold Cross Validation Accuracy")
    dim = np.arange(1, 6, 1)
    plt.xticks(dim)
    plt.show()


def plot_loss(loss):
    loss = abs(loss)
    plt.xlabel('Fold')
    plt.ylabel('Loss')
    plt.plot([1, 2, 3, 4, 5], loss)
    plt.title("5-Fold Cross Validation Loss")
    dim = np.arange(1, 6, 1)
    plt.xticks(dim)
    plt.show()


data_locationF = pathlib.Path('./datasets/pngF').glob('./*.png')
data_locationN = pathlib.Path('./datasets/pngN').glob('./*.png')
data_locationS = pathlib.Path('./datasets/pngS').glob('./*.png')
data_locationV = pathlib.Path('./datasets/pngV').glob('./*.png')
data_sortedF = sorted([x for x in data_locationF])
data_sortedN = sorted([x for x in data_locationN])
data_sortedS = sorted([x for x in data_locationS])
data_sortedV = sorted([x for x in data_locationV])

data_list = data_sortedF + data_sortedN + data_sortedS + data_sortedV
labels = list(['F', 'N', 'S', 'V'])
X = []
y = []

for filename in data_list:
    X.append(load_image(filename))
for idx in range(len(data_sortedF)):
    y.append('F')
for idx in range(len(data_sortedN)):
    y.append('N')
for idx in range(len(data_sortedS)):
    y.append('S')
for idx in range(len(data_sortedV)):
    y.append('V')


print('dataset target names')
print(labels)

X = np.array(X)
y = np.array(y)
dataset_size = len(X)

X = X.reshape(dataset_size, -1)
X = pd.DataFrame(X)
y = pd.DataFrame(y)

dataset = pd.concat((X, y), axis=1)

plt.plot()