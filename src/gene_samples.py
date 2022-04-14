import numpy as np
import sklearn.model_selection


def gene_sam(alpha = [0.25,0.25,0.25,0.25]):
    """
    生成k（自行确定）个满足高斯分布的类数据，每个类有独立的均值和方差，并且根据
    :param alpha: 各类的成分系数
    :return:X_train,X_test with dim (2,n_train) 、 (2，n_test)
    """
    total_num = 400
    #means
    mean_CLass_1 = [0.9,0.8]
    mean_CLass_2 = [-0.8,-0.6]
    mean_CLass_3 = [-0.7,0.8]
    mean_CLass_4 = [0.9,-0.8]
    #covariances
    cov_Cla_1 = 0.05
    cov_Cla_2 = 0.04
    cov_Cla_3 = 0.02
    cov_Cla_4 = 0
    #generate X with (m,n) dim,where m = 2 n is the number of the samples
    sam_C1 = np.random.multivariate_normal\
        (mean = mean_CLass_1,cov =[[0.11,cov_Cla_1],[cov_Cla_1,0.12]],\
         size = int(total_num * alpha[0])).T
    sam_C2 = np.random.multivariate_normal \
        (mean=mean_CLass_2, cov=[[0.14, cov_Cla_2], [cov_Cla_2, 0.1]], \
         size=int(total_num * alpha[1])).T
    sam_C3 = np.random.multivariate_normal \
        (mean=mean_CLass_3, cov=[[0.05, cov_Cla_3], [cov_Cla_3, 0.1]], \
         size=int(total_num * alpha[2])).T
    sam_C4 = np.random.multivariate_normal \
        (mean=mean_CLass_4, cov=[[0.12, cov_Cla_4], [cov_Cla_4, 0.1]], \
         size=int(total_num * alpha[3])).T
    X = np.c_[sam_C1,sam_C2]
    X = np.c_[X,sam_C3]
    X = np.c_[X,sam_C4]
    X_train,X_test = sklearn.model_selection.train_test_split(X.T,test_size= 0.2)

    return X_train.T,X_test.T


