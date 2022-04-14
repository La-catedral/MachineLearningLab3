import numpy as np
import k_means
import itertools


def calc_acc(mean,X,y,T):
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
    print(T)
    print(y)
    for a_perm in all_perm:
        count = 0
        for i in range(samp_num):
            if a_perm[y[i]] == T[i,0]:
                count += 1
        all_acc.append(count)
    return np.max(all_acc) * 1.0 / samp_num





def random_get_sigma(k,m):
    a = []
    for i in range(k):
        X = np.random.rand(m,m)
        X = np.diag(X.diagonal())
        a.append(X)
    return np.array(a)

def normal_prob(x, mu, sigma):
    """
    给定x，高斯分布的参数，计算概率密度（pdf的造轮子实现）
    :param x: (m,1)
    :param mu: array with (m,)
    :param sigma: (m,m)
    :return:
    """
    m = x.shape[0]
    sigma_inv = np.linalg.inv(sigma)
    x = np.array([x]).T  # 将x、mu转为列向量的标准格式
    mu_vec = np.array([mu]).T
    frac_s = np.exp(-1 / 2 * (x - mu_vec).T.dot(sigma_inv).dot(x - mu_vec))  # 求出概率密度函数的分子
    frac_m = (2 * np.pi) ** (m / 2) * np.linalg.det(sigma) ** (1 / 2)  # 求出概率密度的分母

    return frac_s / frac_m


def calc_likelihood(X, mu, sigma, alpha):
    """
    计算给定数据和参数的似然值
    :param X: (m,n) 需要计算似然函数值的训练集
    :param mu: （k,m）各个类的均值构成的矩阵
    :param sigma: (k,m,m) 各类协方差矩阵构成的张量
    :param alpha: (1,k) 各类分布概率构成的行向量
    :return: 给定参数与数据对应的似然函数值
    """
    samp_num = X.shape[1]  # 记录样本数
    k = alpha.shape[1]  # 记录类数
    total_sum = 0  # 存储返回结果的变量
    for i in range(samp_num):
        cla_sum = 0
        for l in range(k):
            cla_sum += alpha[0][l] * normal_prob(X[:, i], mu[l], sigma[l])
        total_sum += np.log(cla_sum)  # 对每个样本的分布概率密度的对数进行累加
    return total_sum


def gaussian_mix(X_train, k, delta=1e-13):
    """
    Gaussian Mixture Model
    :param X_train: (m,n) matrix,where m is the dim of a sample's vector
    :param k: the number of classes
    :param delta: used to justify whether should terminate the loop
    :return:
    """
    size = X_train.shape[1]  # 样本数
    m = X_train.shape[0]  # 特征维度
    alpha = np.random.dirichlet(np.ones(k), size=1)  # 初始化alpha ，(1,4) np array
    mu = k_means.k_means_func(X_train, k)  # 采用k-means 进行初始化  (k,m)
    sigma = random_get_sigma(k,m)

    while True:
        gamma = np.zeros((size, k),dtype=float)  # calculate gamma with current parameters:the expectation of EM
        likelihood = calc_likelihood(X_train, mu, sigma, alpha)
        print("likelihood = " + str(float(likelihood)))
        for i in range(size):  # calc E pace
            x = X_train[:, i]
            sum = 0  # calculate the probability of sample j
            for l in range(k):
                sum += alpha[0][l] * normal_prob(x, mu[l], sigma[l])
            for j in range(k):
                gamma[i][j] = alpha[0][j] * normal_prob(x, mu[j], sigma[j]) / sum
        new_alpha = np.zeros(alpha.shape,dtype=float)
        new_mu = np.zeros(mu.shape, dtype=float)
        new_sigma = np.zeros(sigma.shape,dtype=float)
        for i in range(k):  # calc M pace
            sum_of_gamma = 0
            for j in range(size):
                sum_of_gamma += gamma[j][i]
            new_alpha[0][i] = sum_of_gamma / size

            sum_frac_son_mu = np.zeros((1, m),dtype=float)
            for j in range(size):
                sum_frac_son_mu += gamma[j][i] * X_train[:, j].T
            new_mu[i] = sum_frac_son_mu / sum_of_gamma

            sum_frac_son_sigma = np.zeros((m, m),dtype=float)
            for j in range(size):
                x = np.array([X_train[:,j]]).T
                mu_d = np.array([new_mu[i]]).T
                sum_frac_son_sigma += gamma[j][i] * np.dot(x-mu_d , (x-mu_d).T)
            new_sigma[i] = sum_frac_son_sigma / sum_of_gamma

        this_delt = np.linalg.norm(new_alpha - alpha) + np.linalg.norm(new_mu - mu) + \
                    np.sum([np.linalg.norm(new_sigma[i] - sigma[i]) for i in range(k)])
        if this_delt < delta:  # 如果变化量小于设定阈值，停止迭代
            break
        else:  # 否则更新M步得到的参数
            alpha = new_alpha
            mu = new_mu
            sigma = new_sigma

    y = []
    for i in range(size):  # 重新计算最终的gamma
        x = X_train[:, i]
        min_prob = 0
        min_ind = 0
        sum = 0  # calculate the probability of sample j
        for l in range(k):
            sum += alpha[0][l] * normal_prob(x, mu[l], sigma[l])
        for j in range(k):
            gamma[i][j] = alpha[0][j] * normal_prob(x, mu[j], sigma[j]) / sum
            if gamma[i][j] > min_prob:
                min_prob = gamma[i][j]
                min_ind = j
        y.append(min_ind)

    return mu,y  # 返回各个 聚类中心



