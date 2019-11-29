import numpy as np
import pandas as pd
import pathlib
import os

file_location = 'C:\\Users\\wldns\\PycharmProjects\\ecg-image-classifier\\datasets'

F_files = os.listdir('./datasets/pngF/')
N_files = os.listdir('./datasets/pngN/')
S_files = os.listdir('./datasets/pngS/')
V_files = os.listdir('./datasets/pngV/')

count = 0
for name in F_files:
    newname = name[4:]
    count += 1
    os.rename(file_location+'\pngF\\'+name, file_location+'\pngF\\'+newname)

count = 0
for name in N_files:
    newname = name[4:]
    count += 1
    os.rename(file_location+'\pngN\\'+name, file_location+'\pngN\\'+newname)

count = 0
for name in S_files:
    newname = name[4:]
    count += 1
    os.rename(file_location+'\pngS\\'+name, file_location+'\pngS\\'+newname)

count = 0
for name in V_files:
    newname = name[4:]
    count += 1
    os.rename(file_location+'\pngV\\'+name, file_location+'\pngV\\'+newname)