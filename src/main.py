import k_means
import Gaussian_mix
import gene_samples
import matplotlib.pyplot as plt
import read_data


def hand_gene_samples():
    """
    手动生成数据并观测k-means和GMM结果
    :return:
    """
    k = 4
    X_train,X_test = gene_samples.gene_sam()
    means_1,y = Gaussian_mix.gaussian_mix(X_train,k)  # (k,m)
    means_2 = k_means.k_means_func(X_train,k)
    print("result of k-means: ")
    print(means_2)
    print()
    print("result of GMM: ")
    print(means_1)
    print()
    plt.figure(1)
    plt.scatter(X_train[0,:],X_train[1,:],marker = '.')
    for i in range(k):
        plt.scatter(means_1[i][0],means_1[i][1],marker='*')
    plt.title("GMM")
    plt.figure(2)
    plt.scatter(X_train[0,:],X_train[1,:],marker = '.')
    for i in range(k):
        plt.scatter(means_2[i][0],means_2[i][1],marker='*')
    plt.title("k-means")
    plt.show()
    return


def test_on_iris():
    X, Y = read_data.iris_data_load()  # 使用iris做训练集
    means,y = Gaussian_mix.gaussian_mix(X,3 )  # (k,m)
    print("the GMM's accuracy for iris is :"+ str(Gaussian_mix.calc_acc(means,X,y,Y)))
    means_2 = k_means.k_means_func(X,3)
    print("the k-means' accuracy for iris is :"+ str(k_means.calc_acc(means_2,X,Y)))
    return

def test_on_seeds():
    X,Y =  read_data.seeds_data_load()  # 使用seeds作训练集
    means,y = Gaussian_mix.gaussian_mix(X,2 )  # (k,m)
    print("the GMM's accuracy for seeds is :"+ str(Gaussian_mix.calc_acc(means,X,y,Y)))
    means_2 = k_means.k_means_func(X, 2)
    print("the k-means' accuracy for seeds is :" + str(k_means.calc_acc(means_2, X, Y)))
    return


# hand_gene_samples()
# test_on_iris()
test_on_seeds()