from __future__ import division
from __future__ import print_function

import gzip
import numpy as np
if __name__ == '__main__':
    import data_dirs
else:
    from tools import data_dirs
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

DATADIR = data_dirs.waste

NUM_LABELS = 5
IMAGE_SHAPE = [277, 277, 3]

def get_data():
    """Utility for convenient data loading."""
    X = np.load(DATADIR + '/X.npy').astype('uint8')
    Y = load_label()
    return train_test_split(X, Y, test_size=0.8, random_state=42)
    # return X, X, Y, Y

def load_label():
    labels = np.load(DATADIR + '/Y.npy')
    arr = np.arange(1,6)
    return labels.dot(arr).astype('uint8')


def main():
    Xtrain,Xtest, Ytrain, Ytest = get_data()
    print(Xtrain.shape)    
    print(Xtest.shape)

if __name__ == '__main__':
    main()