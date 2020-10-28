# coding=utf-8
# import sys
# sys.path.append(".../EM/")
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from SW import sw
from OPM import OPM

epsilon_list = [0.5,1, 1.5, 2, 2.5, 3]


def varingepsilon():
    binnumber = 1024
    experimentTimes = 50
    ori = np.array(generate_data())
    n = ori.size
    ns_nist, _ = np.histogram(ori, bins=binnumber, range=(-1, 1))
    ori_formal = ns_nist / n
    accuracy_wd_1 = []
    accuracy_wd_2 = []
    accuracy_range_1 = []
    accuracy_range_2 = []
    accuracy_median_1 = []
    accuracy_median_2 = []
    for epsilon in epsilon_list:
        acc1 = 0
        acc2 = 0
        acc1_range = 0
        acc2_range = 0
        acc1_median = 0
        acc2_median = 0
        print("epsilon is \t ", epsilon)
        for ex_times in range(experimentTimes):
            theta_OPM = OPM(np.array(ori), epsilon, randomized_bins=binnumber, domain_bins=binnumber)
            theta_OPM = theta_OPM / n
            theta_sw = sw(np.array(ori), 0, 1, epsilon, randomized_bins=binnumber, domain_bins=binnumber)
            theta_sw = theta_sw / n
            wda, wdb = compareWasserstein_distances(ori_formal, theta_OPM, theta_sw)
            median_a, median_b = comparemedian(ori_formal, theta_OPM, theta_sw)
            range_a, range_b = rangeQuery(ori_formal, theta_OPM, theta_sw)
            print(wda, wdb)
            acc1 += wda
            acc2 += wdb
            acc1_range += range_a
            acc2_range += range_b
            acc1_median += median_a
            acc2_median += median_b
        accuracy_wd_1.append(acc1 / experimentTimes)  # wd 距离
        accuracy_wd_2.append(acc2 / experimentTimes)  # wd 距离
        accuracy_range_1.append(acc1_range / experimentTimes)  # range query
        accuracy_range_2.append(acc2_range / experimentTimes)  # range query
        accuracy_median_1.append(acc1_median / experimentTimes)  # median
        accuracy_median_2.append(acc2_median / experimentTimes)  # median

    print(accuracy_wd_1)
    print(accuracy_wd_2)
    print(accuracy_range_1)
    print(accuracy_range_2)
    print(accuracy_median_1)
    print(accuracy_median_2)
    return accuracy_wd_1,accuracy_wd_2,accuracy_range_1,accuracy_range_2,accuracy_median_1,accuracy_median_2

def rangeQuery(ori, a, b):
    # 设置 10个任意区间
    length = ori.size
    alpha = 0.1
    acc1 = 0
    acc2 = 0
    for i in range(5):
        left = random.random() * (1 - alpha)
        right = int((left + alpha) * length)
        left = int(left * length)
        sum_ori = np.sum(ori[left:right])
        sum_a = np.sum(a[left:right])
        sum_b = np.sum(b[left:right])
        acc1 += abs(sum_ori - sum_a)
        acc2 += abs(sum_ori - sum_b)

    alpha = 0.4
    for i in range(5):
        left = random.random() * (1 - alpha)
        right = int((left + alpha) * length)
        left = int(left * length)
        sum_ori = np.sum(ori[left:right])
        sum_a = np.sum(a[left:right])
        sum_b = np.sum(b[left:right])
        acc1 += abs(sum_ori - sum_a)
        acc2 += abs(sum_ori - sum_b)
    acc1 = acc1 / 10
    acc2 = acc2 / 10
    return acc1, acc2


def comparemedian(ori, a, b):
    length = ori.size
    accurcy1 = 0  # ori 和 a 对比的数据结果
    accurcy2 = 0  # ori 和 b 对比的数据结果
    exam_value = [0.25, 0.5, 0.75]
    for value in exam_value:
        sumalla = 0
        i = 0
        while sumalla < value and i < length:
            sumalla += ori[i]
            i += 1
        median_ori = i
        sumalla = 0
        i = 0
        while sumalla < value and i < length:
            sumalla += a[i]
            i += 1
        median_a = i
        sumalla = 0
        i = 0
        while sumalla < value and i < length:
            sumalla += b[i]
            i += 1
        median_b = i
        accurcy1 += abs(median_a - median_ori)
        accurcy2 += abs(median_ori - median_b)
    accurcy1 = accurcy1 / len(exam_value)
    accurcy2 = accurcy2 / len(exam_value)
    return accurcy1, accurcy2


def compareWasserstein_distances(ori, a, b):
    # print(sum(b))
    wda = 0
    wdb = 0
    for i, j in zip(ori, a):
        wda += abs(i - j)
    for i, j in zip(ori, b):
        wdb += abs(i - j)
    return wda, wdb


def generate_data():
    res = []
    a = np.random.normal(0, math.sqrt(0.02), 50000 * 5)
    res.extend(a)
    b = np.random.normal(0.2, math.sqrt(0.05), 50000 * 3)
    res.extend(b)
    c = np.random.normal(-0.6, math.sqrt(0.02), 50000 * 2)
    res.extend(c)
    new_res = []
    for i in res:
        if i <= 1 and i >= -1:
            new_res.append(i)
    return new_res


if __name__ == "__main__":
    varingepsilon()

    # binnumber = 1024
    # ori = generate_data()
    # ns_nist, _ = np.histogram(ori, bins=binnumber, range=(-1, 1))
    #
    # theta = OPM(np.array(ori), 1, randomized_bins=binnumber, domain_bins=binnumber)
    # print(theta)
    # x = [i for i in range(len(theta))]
    # plt.bar(x, theta)
    # plt.xlabel(r'$\epsilon$', fontsize=16)
    # plt.ylabel(u'2', fontsize=20)
    # plt.tick_params(labelsize=20)
    # # plt.text(1.19, 3.5, "key point", fontdict={'size': '16', 'color': 'b'})
    # # plt.title(u"Comparision and varying epsilon", fontsize=16)
    # # plt.axvline(x=1.185, ls="-", c="red")  # 添加垂直直线
    # # plt.axvline(x=1.29, ls="-", c="gray")  # 添加垂直直线
    # plt.legend()
    # plt.show()
