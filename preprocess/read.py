import pickle
from os import listdir, path

with open('y_test.pkl','rb') as f:
    data = pickle.load(f)

print(len(data))