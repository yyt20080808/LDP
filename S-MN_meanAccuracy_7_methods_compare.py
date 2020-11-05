# coding=utf-8
# import sys
# sys.path.append(".../EM/")
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from SW import sw,swNos
from OPM import OPM,OPM_meanonly,OPMnoS

epsilon_list = [0.5,1,1.5,2,2.5,3]
def generate_data2():
    res = []
    a = np.random.normal(0.3,0.4,30000)
    res.extend(a)
    a = np.random.normal(-0.2,0.3,30000)
    res.extend(a)

    new_res = []
    for i in res:
        if i <= 0.9 and i >= -0.85:
            new_res.append(i)
    ns_hist, _ = np.histogram(np.array(new_res), bins=512, range=(-1,1))
    plt.bar([i for i in range(512)],ns_hist)
    plt.show()
    return new_res

def meanCompare():
    binnumber = 512
    experimentTimes = 40
    ori = np.array(generate_data2())
    n = ori.size
    true_mean = np.sum(ori)/n
    print("真实均值：",true_mean)
    accuracy_mean_1 = []
    accuracy_mean_2 = []
    accuracy_mean_3 = []
    accuracy_mean_4 = []
    accuracy_mean_5 = []
    accuracy_mean_6 = [] # OPM 和 OPM smooth  加起来除以 2
    accuracy_mean_7 = [] # OPM 和 OPM nos 加起来除以2
    for epsilon in epsilon_list:
        acc1 = 0
        acc2 = 0
        acc3 = 0
        acc4 = 0
        acc5 = 0
        acc6 = 0
        acc7 = 0
        print("epsilon is \t",epsilon)

        for ex_times in range(experimentTimes):
            theta_sw_NoS = swNos(np.array(ori), epsilon, randomized_bins=binnumber, domain_bins=binnumber)
            theta_sw_NoS = theta_sw_NoS / n
            sum_swnos = 0
            for i in range(binnumber):
                sum_swnos += theta_sw_NoS[i] * (i + 0.5)
            sum_swnos = (sum_swnos / binnumber - 0.5) * 2  # 转换到 -1 到 1 的区间

            theta_sw = sw(ori, 0, 1, epsilon, randomized_bins=binnumber, domain_bins=binnumber)
            theta_sw = theta_sw/n
            sum_sw = 0
            for i in range(binnumber):
                sum_sw += theta_sw[i] * (i+0.5)
            sum_sw = (sum_sw/binnumber -0.5) * 2 # 转换到 -1 到 1 的区间
            OPM_mean = OPM_meanonly(ori,epsilon) / n

            theta_OPM_NoS = OPMnoS(np.array(ori), epsilon, randomized_bins=binnumber, domain_bins=binnumber)
            theta_OPM_NoS = theta_OPM_NoS / n
            sum_OPMnos = 0
            for i in range(binnumber):
                sum_OPMnos += theta_OPM_NoS[i] * (i + 0.5)
            sum_OPMnos = (sum_OPMnos / binnumber - 0.5) * 2  # 转换到 -1 到 1 的区间

            theta_OPM = OPM(ori, epsilon, randomized_bins=binnumber, domain_bins=binnumber)
            theta_OPM = theta_OPM / n
            sum_OPM = 0
            for i in range(binnumber):
                sum_OPM += theta_OPM[i] * (i + 0.5)
            sum_OPM = (sum_OPM / binnumber - 0.5) * 2

            acc1 += abs(sum_sw - true_mean)
            acc2 += abs(OPM_mean - true_mean)
            acc3 += abs(sum_swnos - true_mean)
            acc4 += abs(sum_OPMnos - true_mean)
            acc5 += abs(sum_OPM - true_mean)
            acc6 += abs((OPM_mean + sum_OPMnos)/2 - true_mean)
            acc7 += abs((OPM_mean + sum_OPM)/2 - true_mean)
            print(abs(sum_sw - true_mean))
            print("OPM\t",abs(OPM_mean - true_mean))
            print(abs(sum_swnos - true_mean))
            print("OPM ems nos\t", abs(sum_OPMnos - true_mean))
            print("OPM ems\t",abs(sum_OPM - true_mean))
            print("add together no smooth\t",abs((OPM_mean + sum_OPMnos)/2 - true_mean))
            print("add together smooth\t", abs((OPM_mean + sum_OPM)/2 - true_mean))
        accuracy_mean_1.append(acc1 / experimentTimes)  # wd 距离
        accuracy_mean_2.append(acc2 / experimentTimes)
        accuracy_mean_3.append(acc3 / experimentTimes)
        accuracy_mean_4.append(acc4 / experimentTimes)
        accuracy_mean_5.append(acc5 / experimentTimes)
        accuracy_mean_6.append(acc6 / experimentTimes)
        accuracy_mean_7.append(acc7 / experimentTimes)
    print("sw\t",accuracy_mean_1)
    print("OPM\t",accuracy_mean_2)
    print("swNos\t", accuracy_mean_3)
    print("OPM + em +smooth\t", accuracy_mean_4)
    print("OPM + em\t", accuracy_mean_5)
    print("mix + em \t", accuracy_mean_6)
    print("mix + em+smooth\t", accuracy_mean_7)
if __name__ == "__main__":
    meanCompare()

