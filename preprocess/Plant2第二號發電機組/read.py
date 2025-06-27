import pickle
import os

folder = r'80%Training & 20%Testing' # 20%Training & 80%Testing # 80%Training & 20%Testing
with open(os.path.join('.', folder, 'X_train.pkl'),'rb') as f:
    data = pickle.load(f)

print(len(data))