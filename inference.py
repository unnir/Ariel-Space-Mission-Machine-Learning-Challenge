import os

import numpy as np
from sklearn.externals import joblib
from keras.models import load_model

from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-m", "--model", 
                    help="best model")

args = parser.parse_args()


# test data load 
X_test = np.load('.data/all_test.npy')
X_test_head = np.load('.data/all_test_heads.npy')
sk2 = joblib.load("scaler.save") 
X_test_head = sk2.transform(X_test_head)

X_test = X_test.reshape(-1,55*300)
X_test = (X_test-1)*1000
X_test = X_test.reshape(-1,55,300)


# inference 
y_pred = np.zeros((X_test.shape[0], 55))

if args.model == 'yes':
    PATH_WEIGHTS = './best_weights' 
else:
    PATH_WEIGHTS = './weights' 

for subdir, dirs, files in os.walk(PATH_WEIGHTS):
            for file in tqdm(files):
                model = load_model(whts_file)
                y_pred+= model.predict([X_test,X_test_head])

y_pred /= len(files)
print("Done!")

# save estimations
np.savetxt('sub/y_hat.csv', y_pred/1000)