import numpy as np
import itertools

def calc_acc(mean,X,T):
    """
    计算准确率
    :param mean: (k,m)
    :param X: (m,n)
    :param T: (n,1)
    :return:
    """
    samp_num = X.shape[1]
    k = mean.shape[0]  # 类别数
    all_perm = list(itertools.permutations(range(k)))
    all_acc = []
    y = []
    for i in range (samp_num):  # 求得每个样本的分类
        min_dis = np.linalg.norm(X[:,i] - mean[0])
        min_ind = 0
        for j in range(1,k):
            dis = np.linalg.norm(X[:,i] - mean[j])
            if dis < min_dis:
                min_dis = dis
                min_ind = j
        y.append(min_ind)

    for a_perm in all_perm:
        count = 0
        for i in range(samp_num):
            if a_perm[y[i]] == T[i,0]:
                count += 1
        all_acc.append(count)
    return np.max(all_acc) * 1.0 / samp_num


def k_means_func(X_train,k,random_init = True):
    """
    通过k-means无监督学习出聚类模型
    :param X_train: training data
    :param k: number of classes
    :param random_init: whether give a random init parameter or not
    :return: (k,m) np array
    """
    sam_size = X_train.shape[1]
    #initialization
    if random_init:
        means = []
        for i in range(k):
            means.append(X_train.T[(sam_size//k)*i])

    # 为每个类提供存放该类样本点的容器（集合构成的列表）
    Cla = []
    for i in range(k):
        Cla.append(list()) # 生成k个类

    # 计算每个点到四个类中心的距离
    while True:
        for i in range(k):
            Cla[i].clear()
        for sam_point in X_train.T:
            min_dis = (sam_point[0] - means[0][0])**2 + (sam_point[1] - means[0][1])**2
            min_cla = 0
            for j in range(k):
                this_dis = (sam_point[0] - means[j][0])**2 + (sam_point[1] - means[j][1])**2
                if this_dis < min_dis:
                    min_dis = this_dis
                    min_cla = j
            Cla[min_cla].append(sam_point)

        # 重新计算中心
        means_changed = False
        for i in range(k):
            # 计算k类均值：
            this_mean = np.zeros(X_train.shape[0],dtype=float)
            for point in Cla[i]:
                this_mean += point
            this_mean /= len(Cla[i])
            if not (np.array_equal(this_mean,means[i])):
                means_changed = True
                means[i] = this_mean
        if not means_changed:
            break

    return np.array(means)