from __future__ import division
from __future__ import print_function

import gzip
import numpy as np
import data_dirs

DATADIR = data_dirs.waste

NUM_LABELS = 5
IMAGE_SHAPE = [277, 277, 3]


def get_data(name):
  """Utility for convenient data loading."""
  if name == 'train':
    return np.load(DATADIR + '/X.npy'), load_label()

def load_label():
  labels = np.load(DATADIR + '/Y.npy')
  arr = np.arange(1,6)
  return labels.dot(arr)


def main():
  image, label = get_data('train')
  print(image.shape)
  print(label.shape)

if __name__ == '__main__':
  main()