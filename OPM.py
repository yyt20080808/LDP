# coding=utf-8
import numpy as np

# from scipy.stats import binom
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random

import math


def OPM(ori_samples, eps, randomized_bins=1024, domain_bins=1024):
    e_epsilon = math.e ** eps
    e_epsilon_sqrt = math.sqrt(e_epsilon)
    B = (e_epsilon_sqrt + 9) / (10 * e_epsilon_sqrt - 10)
    k = B * e_epsilon / (B * e_epsilon - 1 - B)
    C = k + B
    q = 1 / (2 * B * e_epsilon + 2 * k)
    p = e_epsilon / (2 * B * e_epsilon + 2 * k)
    pr = 2 * B * e_epsilon * q
    samples = ori_samples
    # randoms = np.random.uniform(0, 1, len(samples)) # 跟数据总量一样多的 概率随机数

    # noisy_samples = np.zeros_like(samples) # 扰动后的数据？

    # report
    noisy_samples = OPM_noise(samples, k, B, C, pr)

    # report matrix
    m = randomized_bins
    n = domain_bins
    m_cell = (2 * C) / m  # 输出空间转化为bin， 每个bin 的size
    n_cell = 2 / n  # 输入空间转化为 bin， 每个bin的size

    transform = np.ones((m, n)) * q * m_cell
    q_value = transform[0, 0]
    p_value = q_value * e_epsilon
    pro_pass = 0
    left_index = 0
    right_index = 0
    # 计算 一列中多少个
    for i in range(0, n):
        if i== n-1:
            sss =1
        if i == 0:
            a = int(B / C * n)
            reseverytime = 1 - a * p_value - (n - a) * q_value
            sss = (p_value-q_value-reseverytime)
            pro_pass = ((n - a-1)  * (p_value - q_value) + (p_value-q_value-reseverytime)) / (n-1)
            transform[0:a, i ] = p_value
            transform[a, i ] = q_value + reseverytime
            right_index = a
            left_index = 0
        else:
            temp_right = transform[left_index,i-1] - pro_pass
            if temp_right >= q_value:
                transform[left_index, i] = transform[left_index,i-1] - pro_pass # 左边减去 1
                if transform[right_index, i-1] + pro_pass < p_value:
                    transform[right_index, i ] = transform[right_index, i-1] + pro_pass
                else: # 说明要 right_index+1
                    overflow = transform[right_index, i-1] + pro_pass - p_value
                    transform[right_index, i] = p_value
                    if right_index >= n-1 and   overflow < 1e-5:
                        transform[left_index + 1:right_index, i] = p_value
                        break
                    right_index+=1
                    transform[right_index,i] = q_value + overflow
            else:
                overflow = pro_pass - (transform[left_index,i-1] - q_value)
                transform[left_index, i] = q_value
                left_index+= 1
                transform[left_index,i] = p_value - overflow
                if transform[right_index, i-1] + pro_pass < p_value:
                    transform[right_index, i ] = transform[right_index, i-1] + pro_pass
                else: # 说明要 right_index+1
                    overflow = transform[right_index, i-1] + pro_pass - p_value
                    transform[right_index, i] = p_value
                    if right_index == n-1 and  overflow < 1e-4:
                        transform[left_index+1:right_index, i] = p_value
                        break
                    right_index+=1
                    transform[right_index,i] = q_value + overflow
            for jjj in range(left_index+1,right_index):
                transform[jjj, i] = p_value
        # print(sum(transform[:, i]))


    # print( np.sum(transform, axis=0)) # 表示每一列每一列。
    # for jjjj in range(n):
    print(transform[:, -2])
    print(transform[:, -1])
    max_iteration = 10000
    loglikelihood_threshold = 1e-3
    ns_hist, _ = np.histogram(noisy_samples, bins=randomized_bins, range=(-1 * C, C))
    # print("noisesss ")
    # print(ns_hist)
    # return EM(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(ori_samples)
    return EMS(m, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(ori_samples)


def OPM_noise(samples, k, B, C, pr):
    res = []
    count = 0
    for value in samples:
        perturbed_value = -1000
        lt = k * value - B
        rt = lt + 2 * B
        # 回答很好的条件
        if random.random() <= pr:
            perturbed_value = random.random() * 2 * B + lt
        else:  # 回答比较差的条件
            temp = random.random()
            ppp = (lt + C) / (2 * k)
            if ppp > temp:
                perturbed_value = temp * (2 * k) - C
            else:
                perturbed_value = rt + (temp - ppp) * (2 * k)

        hhh = 2 * C / 100 - C
        if perturbed_value < hhh:
            count += 1
        res.append(perturbed_value)

    return np.array(res)


def EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold):
    # smoothing matrix
    smoothing_factor = 2
    # binomial_tmp = [binom(smoothing_factor, k) for k in range(smoothing_factor + 1)]
    binomial_tmp = [1, 2, 1]  # [1,6,15,20,15,6,1] #[1, 2, 1]
    smoothing_matrix = np.zeros((n, n))
    central_idx = int(len(binomial_tmp) / 2)
    for i in range(int(smoothing_factor / 2)):
        smoothing_matrix[i, : central_idx + i + 1] = binomial_tmp[central_idx - i:]
    for i in range(int(smoothing_factor / 2), n - int(smoothing_factor / 2)):
        smoothing_matrix[i, i - central_idx: i + central_idx + 1] = binomial_tmp
    for i in range(n - int(smoothing_factor / 2), n):
        remain = n - i - 1
        smoothing_matrix[i, i - central_idx + 1:] = binomial_tmp[: central_idx + remain]
    row_sum = np.sum(smoothing_matrix, axis=1)
    smoothing_matrix = (smoothing_matrix.T / row_sum).T

    # EMS
    theta = np.ones(n) / float(n)
    theta_old = np.zeros(n)
    r = 0
    sample_size = sum(ns_hist)
    old_logliklihood = 0

    while LA.norm(theta_old - theta, ord=1) > 1 / sample_size and r < max_iteration:
        theta_old = np.copy(theta)
        X_condition = np.matmul(transform, theta_old)

        TMP = transform.T / X_condition

        P = np.copy(np.matmul(TMP, ns_hist))
        P = P * theta_old

        theta = np.copy(P / sum(P))

        # Smoothing step
        theta = np.matmul(smoothing_matrix, theta)
        theta = theta / sum(theta)

        logliklihood = np.inner(ns_hist, np.log(np.matmul(transform, theta)))
        imporve = logliklihood - old_logliklihood

        if r > 1 and abs(imporve) < loglikelihood_threshold:
            print("stop when", imporve / old_logliklihood, loglikelihood_threshold)
            break

        old_logliklihood = logliklihood

        r += 1
    # print(imporve)
    return theta


def EM(n, ns_hist, transform, max_iteration, loglikelihood_threshold):
    theta = np.ones(n) / float(n)
    theta_old = np.zeros(n)
    r = 0
    sample_size = sum(ns_hist)
    old_logliklihood = 0

    while LA.norm(theta_old - theta, ord=1) > 1 / sample_size and r < max_iteration:
        theta_old = np.copy(theta)
        X_condition = np.matmul(transform, theta_old)

        TMP = transform.T / X_condition

        P = np.copy(np.matmul(TMP, ns_hist))
        P = P * theta_old

        theta = np.copy(P / sum(P))

        logliklihood = np.inner(ns_hist, np.log(np.matmul(transform, theta)))
        imporve = logliklihood - old_logliklihood

        if r > 1 and abs(imporve) < loglikelihood_threshold:
            # print("stop when", imporve, loglikelihood_threshold)
            break

        old_logliklihood = logliklihood

        r += 1
    return theta


if __name__ == "__main__":
    # ori = np.random.normal(-0.3, 0.3, 50000)
    binnumber = 200
    ori = np.random.uniform(-1, 1, 500000)
    ns_nist, _ = np.histogram(ori, bins=binnumber, range=(-1, 1))
    print(ns_nist)
    # x = [i for i in range(256)]
    # plt.bar(x, ns_nist)
    # plt.show()
    new_ori = []
    for i in ori:
        if i > 1 or i < -1:
            pass
        else:
            new_ori.append(i)
    # for i in ori2:
    #     new_ori.append(i)
    new_ori = np.array(new_ori)
    theta = OPM(new_ori, 1, randomized_bins=binnumber, domain_bins=binnumber)
    print(theta)
    x = [i for i in range(len(theta))]
    plt.bar(x, theta)
    plt.xlabel(r'$\epsilon$', fontsize=16)
    plt.ylabel(u'2', fontsize=20)
    plt.tick_params(labelsize=20)
    # plt.text(1.19, 3.5, "key point", fontdict={'size': '16', 'color': 'b'})
    # plt.title(u"Comparision and varying epsilon", fontsize=16)
    # plt.axvline(x=1.185, ls="-", c="red")  # 添加垂直直线
    # plt.axvline(x=1.29, ls="-", c="gray")  # 添加垂直直线
    plt.legend()
    plt.show()
