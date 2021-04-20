import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import tensorflow as tf


def cv_model_evaluate(interaction_matrix, predict_matrix, train_matrix):
    test_index = np.where(train_matrix == 0)
    real_score = interaction_matrix[test_index]
    predict_score = predict_matrix[test_index]
#     return get_metrics(real_score, predict_score)
#     return calculate_auc(real_score, predict_score)
    return calculate_auc_by_sklearn(real_score, predict_score)


def calculate_auc(y, p):
    '''
    y: 真实Label的向量 ndarray
    p: 预测为正例的概率的向量 ndarray
    '''
    # 保证相同概率的0排在1前面
    print("进入calculate_auc")
    tmp = np.array(sorted(zip(y, p), key=lambda x: (x[1], -x[0]), reverse=True))
    neg = 0
    pos = 0
    for i in y:
        if i == 0:
            neg += 1
        elif i == 1:
            pos += 1
    loss = 0
    neglst = np.array([])
    for i in range(len(tmp)):
        if tmp[i][0] == 1:
            loss += np.sum(neglst == tmp[i][1]) / 2 + np.sum(neglst != tmp[i][1])
        else:
            neglst = np.append(neglst, tmp[i][1])

    return 1 - loss / (neg * pos)


def calculate_auc_by_sklearn(y, p):
    fpr, tpr, _ = roc_curve(y, p)

    result = auc(fpr, tpr)

    return result


def masked_accuracy(preds, labels, mask, negative_mask):
    """Accuracy with masking."""
    print("进入masked_accuracy")
    preds = tf.cast(preds, tf.float32)  
    labels = tf.cast(labels, tf.float32)
    c  =  preds-labels
    print("c:{}".format(c))
    error = tf.square(preds-labels)
    mask += negative_mask
    mask = tf.cast(mask, dtype=tf.float32)
    print("mask:{}".format(mask))
    error *= mask
#     return tf.reduce_sum(error)
    print("preds:{}".format(preds))
    print("labels:{}".format(labels))
    return tf.sqrt(tf.reduce_mean(error))


def euclidean_loss(preds, labels):
    euclidean_loss = tf.sqrt(tf.reduce_sum(tf.square(preds-labels), 0))
    return euclidean_loss





