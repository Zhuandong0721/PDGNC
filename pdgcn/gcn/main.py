import tensorflow as tf
import numpy as np
from train import Training


def div_list(ls,n):
    ls_len=len(ls)
    j = ls_len//n
    ls_return = []
    for i in range(0,(n-1)*j,j):
        ls_return.append(ls[i:i+j])
    ls_return.append(ls[(n-1)*j:])
    return ls_return

if __name__ == "__main__":
  # Initial model
  gcn = Training()
  
  # Set random seed
  seed = 123
  np.random.seed(seed)
  tf.compat.v1.set_random_seed(seed)

  labels = np.loadtxt("data/adj.txt")  
  reorder = np.arange(labels.shape[0])
  np.random.shuffle(reorder)

  cv_num=5

  order = div_list(reorder.tolist(),cv_num)
  for i in range(cv_num):
      print("cross_validation:", '%01d' % (i))
      test_arr = order[i]
      arr = list(set(reorder).difference(set(test_arr)))
      np.random.shuffle(arr)
      train_arr = arr
      scores = gcn.train(train_arr, test_arr)
 
