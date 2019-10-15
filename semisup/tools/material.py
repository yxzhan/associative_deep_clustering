from __future__ import division
from __future__ import print_function
from pathlib import Path
import gzip
import numpy as np
if __name__ == '__main__':
    import data_dirs
else:
    from tools import data_dirs
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

DATADIR = data_dirs.material

NUM_LABELS = 5
IMAGE_SHAPE = [32,32, 3]
base_path = Path(__file__).parent
TEST_SIZE = 0.8

def get_data(one_hot=True):
    """Utility for convenient data loading."""
    X = load_image()
    Y = load_label(one_hot)
    return train_test_split(X, Y, test_size=TEST_SIZE, random_state=42)
    # return X, X, Y, Y

def load_label(one_hot=True):
    labels = np.load((base_path / '../data/npy/Y.npy').resolve())
    if one_hot != True:
        arr = np.arange(0,5)
        return labels.dot(arr).astype('uint8')
    else:
        return labels    


def main():
    # Xtrain,Xtest, Ytrain, Ytest = get_data()
    # print(Xtrain.shape) 
    # print(Xtest.shape)
    a = load_label()
    print(a.shape)

def load_image():    
    images = np.load((base_path / '../data/npy/X_new.npy').resolve()).astype('uint8')
    images_resize = []
    for image in images:
        images_resize.append(resize_img(image))
    return np.array(images_resize).astype('uint8')

def resize_img(img):
    # return img
    return resize(img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]),anti_aliasing=True)*255

if __name__ == '__main__':
    main()


augmentation_params = dict()
augmentation_params['max_crop_percentage'] = 0.2
augmentation_params['brightness_max_delta'] = 0.5
augmentation_params['noise_std'] = 0.05
augmentation_params['flip'] = True
augmentation_params['max_rotate_angle'] = 15