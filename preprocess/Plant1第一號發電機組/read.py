import pickle
from os import listdir, path

with open('X_train','rb') as f:
    data = pickle.load(f)

print(len(data))