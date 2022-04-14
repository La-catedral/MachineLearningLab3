import numpy as np



def iris_data_load():
    data = open('iris.txt').readlines()
    data_set = []
    y_set = []
    for data_line in data :
        this_line = data_line.strip().split(',')
        this_line_float = []
        for i in range(4):
            fl = float(this_line[i])
            this_line_float.append(fl)
        data_set.append(this_line_float)
        y_set.append(this_line[4])  # 读取分类
    data_set = np.array(data_set)
    X = data_set.T
    Y = []
    for str in y_set:
        if str == 'Iris-setosa':
            Y.append(0)
        elif str == 'Iris-versicolor':
            Y.append(1)
        else:
            Y.append(2)
    Y = np.array([Y]).T
    return X,Y

def seeds_data_load():
    data = open('seeds_dataset.txt').readlines()
    data_set = []
    for data_line in data :
        this_line = data_line.strip().split()
        this_line_float = []
        for str in this_line:
            fl = float(str)
            this_line_float.append(fl)
        if this_line_float[7] == 3:
            break
        data_set.append(this_line_float)
    data_set = np.array(data_set)
    X = data_set[:, 0:7]
    Y_raw = data_set[:, 7]
    Y = []
    for i in Y_raw:
        Y.append(int(i))

    Y = np.array([Y]).T
    Y = Y - 1
    print(X)
    print(Y)
    return X.T,Y