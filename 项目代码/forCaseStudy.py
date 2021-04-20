from train import *
import numpy as np
from utils import *
from clac_metric import *
import gc

import codecs

if __name__ == "__main__":
  dis_sim = np.loadtxt("data(383-495)/d-d(diag_zero).csv",
                       delimiter=' ')
  miRNA_sim = np.loadtxt('data(383-495)/m-m_diag_zero.csv',
                         delimiter=' ')

  miRNA_dis_matrix = np.loadtxt('data(383-495)/m-d.csv', delimiter=',')
  index_matrix = np.mat(np.where(miRNA_dis_matrix == 1))
  association_nam = index_matrix.shape[1]

  # seed = 0
  miRNA_dis_matrix, predict_y_proba, train_matrix = train([], RNA_SIMW * miRNA_sim, dis_SIMW * dis_sim)
  predict_y_proba = predict_y_proba.reshape(train_matrix.shape[0], train_matrix.shape[1])

  known = []
  unknown = []
  special = 204
  A = train_matrix.getA()
  for j in range(A.shape[0]):
        if A[j][special] == 1:
           pass;
        else:
            unknown.append([j, special, predict_y_proba[j][special]])

  index = []
  score = []
  for ele in unknown:
      index.append(ele[0])
      score.append(ele[2])
  index = np.array(index)
  # scores = np.array(predict_y_proba[:,special])
  score = np.array(score)
  print(score)
  rank_idx = np.argsort(score)   # 从中可以看出argsort函数返回的是数组值从小到大的索引值
  index = index[rank_idx]
  index = index[-10:]
  print(index)

  miRNA_name = []

  with codecs.open('data(383-495)/miRNA_name.txt', 'r', 'utf-8') as fd:

    for line in fd.readlines():
      miRNA_name.append(list(map(str, line.split(','))))

  for i in index:
     print(miRNA_name[i])
