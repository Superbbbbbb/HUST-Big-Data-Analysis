import numpy as np
import random as rd
import matplotlib.pyplot as plt
import csv


def long_s(center, data):
    s = 0
    for i in range(1, 14):
        s += pow(center[i] - float(data[i]), 2)
    return s


def sse(m, data_min):  # 计算距离平方和SSE
    sse_num = [0, 0, 0]
    for i in range(m):
        sse_num[int(data_min[i, 0]) - 1] += data_min[i, 1]
    sse = sse_num[0] + sse_num[1] + sse_num[2]
    print('SSE总和:', sse)
    return sse


def get_points(m, data, j, data_min):
    point = []
    for i in range(m):
        if data_min[i, 0] == j + 1:  # 根据保存的信息进行分类
            point.append(data[i])
    return point


def cal_acc(m, data, k, data_min):
    hit = 0
    for j in range(k):
        cluster_tmp = [0, 0, 0]
        point = get_points(m, data, j, data_min)
        for item in point:
            cluster_tmp[int(item[0]) - 1] += 1
        hit += max(cluster_tmp)
    acc = hit / len(data)
    print('准确度：', acc)  # 准确度为正确分类数与总数之比


def kmeans(centers):
    data_min = np.mat(np.zeros((m, 2)))
    flag = True
    while flag:
        flag = False
        for i in range(m):
            min_s = 1000000.0  # 初始化最小距离
            min_center = -1
            for j in range(k):
                distance = long_s(centers[j], data[i])
                if distance < min_s:
                    min_s = distance  # 最小值
                    min_center = j + 1  # 对应中心点
            if data_min[i, 0] != min_center or data_min[i, 1] != min_s:
                data_min[i, :] = min_center, min_s  # 更新中心点和距离信息
                flag = True
        for j in range(k):
            point = get_points(m, data, j, data_min)  # 点分类
            centers = np.array(centers)
            centers[j, :] = np.mean(point, axis=0)
    return data_min, centers


if __name__ == '__main__':
    with open('归一化数据.csv') as f:
        reader = csv.reader(f)
        data = []
        for j in reader:
            j = list(map(float, j))
            data.append(j)
    m = len(data)
    k = 3
    centers = []
    for i in range(k):
        c = [0]
        for j in range(13):
            c.append(rd.random())
        centers.append(c)
    data_min, centers = kmeans(centers)

    sse(m, data_min)
    cal_acc(m, data, k, data_min)
    colors = ['r', 'g', 'b']
    for j in range(k):
        point = get_points(m, data, j, data_min)
        point = np.array(point)
        plt.scatter(point[:, 6], point[:, 7], c=colors[j])
    plt.scatter(centers[:, 6], centers[:, 7], marker='*', s=200, c='black')
    plt.show()
    with open("result.csv", 'w') as f:
        for i in range(m):
            f.write("{}\t,{}\n".format(data_min[i, 0], data_min[i, 1]))
