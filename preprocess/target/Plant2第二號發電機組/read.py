import pickle
from os import listdir, path

with open('y_train.pkl','rb') as f:
    data = pickle.load(f)

print(len(data))