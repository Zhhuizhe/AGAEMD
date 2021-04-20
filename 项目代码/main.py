from train import *
from utils import *
from clac_metric import *
import gc
import random
from constant import *
from timeit import default_timer as timer

if __name__ == "__main__":
    tic = timer()
    # seed = 0
    # dis_sim = np.loadtxt("data(383-495)/d-d(diag_zero).csv", delimiter=' ')
    # miRNA_sim = np.loadtxt('data(383-495)/miRNA_Smi_HMDDv2_fromTensor.txt', delimiter=' ')
    #
    # miRNA_dis_matrix = np.loadtxt('data(383-495)/m-d.csv', delimiter=',')
    # 读取miRNA相似性矩阵，读取disease相似性矩阵，读取miRNA-disease关联矩阵
    dis_sim = np.loadtxt("HMDD3.2/HMDDv3.2_from_Tensor_v2/disease_similarity_new2.txt", delimiter=' ')
    miRNA_sim = np.loadtxt('HMDD3.2/HMDDv3.2_from_Tensor_v2/mir_fun_sim_matrix_new2.txt', delimiter=' ')
    miRNA_dis_matrix = np.loadtxt('HMDD3.2/HMDDv3.2_from_Tensor_v2/combine_association_matrix.txt', delimiter=' ')

    repeat_times = 5
    AUC_total = 0
    for i in range(repeat_times):
        print('-------------------------------------------------------------------------------------------------------')
        print('-----------第{}次测试------'.format(i))

        index_matrix = np.mat(np.where(miRNA_dis_matrix == 1))
        association_nam = index_matrix.shape[1]
        random_index = index_matrix.T.tolist()
        random.shuffle(random_index)
        # 将随机打乱后的数据集分为5组，其中0~3组为训练集，第4组为测试集
        k_folds = 5
        # k_folds = 10
        CV_size = int(association_nam / k_folds)  # 2506
        temp = np.array(random_index[:association_nam - association_nam % k_folds]).reshape(k_folds, CV_size, -1).tolist()  # 将二者之间的联系分为5组  5*1086
        temp[k_folds - 1] = temp[k_folds - 1] + random_index[association_nam - association_nam % k_folds:]
        random_index = temp  # 5*2506
        metric = np.zeros((1, 7))
        print("--------输出参数----------------------------------------------------------------------------")
        print("EPOCH:{},EMB_DIM:{},LEARNING_RATE:{},ADJ_DP:{},DROUPOUT:{},RNA_SIMW:{},dis_SIMW:{},VERSION:{}".format(EPOCH,
                                                                                                                 EMB_DIM,
                                                                                                                 LEARNING_RATE,
                                                                                                                 ADJ_DP,
                                                                                                                 DROUPOUT,
                                                                                                                 RNA_SIMW,
                                                                                                                 dis_SIMW,
                                                                                                                 VERSION))
        for k in range(k_folds):
            # 注意该处miRNA和disease相似度矩阵已经乘上了权重因子
            miRNA_dis_matrix, predict_y_proba, train_matrix = train(random_index[k], RNA_SIMW * miRNA_sim,
                                                                dis_SIMW * dis_sim)
            predict_y_proba = predict_y_proba.reshape(train_matrix.shape[0], train_matrix.shape[1])
            metric_tmp = cv_model_evaluate(miRNA_dis_matrix, predict_y_proba, train_matrix)

            print(metric_tmp)
            metric += metric_tmp
            del train_matrix
            gc.collect()
        print(metric / k_folds)
        AUC_total = AUC_total + metric
        metric = np.array(metric / k_folds)

        print("第{}次的最终结果：{}".format(i, metric))
    print("多次运行的最终结果:")
    print(np.array(AUC_total/(k_folds*repeat_times)))
    toc = timer()
    print("总共运行时间：")
    print(toc - tic)  # 输出的时间，秒为单位

