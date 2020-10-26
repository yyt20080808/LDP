#coding=utf-8
import numpy as np

# from scipy.stats import binom
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random

import math
def OPM(ori_samples,  eps, randomized_bins=1024, domain_bins=1024):
    e_epsilon = math.e**eps
    e_epsilon_sqrt = math.sqrt(e_epsilon)
    B = (e_epsilon_sqrt+9) / (10*e_epsilon_sqrt-10)
    k = B * e_epsilon / (B * e_epsilon - 1 - B)
    C = k + B
    q = 1 / (2 * B * e_epsilon + 2 * k)
    p = e_epsilon / (2 * B * e_epsilon + 2 * k)
    pr = 2 * B * e_epsilon * q
    samples = ori_samples
    # randoms = np.random.uniform(0, 1, len(samples)) # 跟数据总量一样多的 概率随机数

    # noisy_samples = np.zeros_like(samples) # 扰动后的数据？

    # report
    noisy_samples= OPM_noise(samples,k,B,C,pr)

    # report matrix
    m = randomized_bins
    n = domain_bins
    m_cell = (2*C) / m  # 输出空间转化为bin， 每个bin 的size
    n_cell = 2 / n   # 输入空间转化为 bin， 每个bin的size

    transform = np.ones((m, n)) * q * m_cell
    q_value = transform[0,0]
    p_value = q_value * e_epsilon
    for i in range(n):
        # 遍历所有bin的值
        # 输入的bin的左右边界
        left_input_v = -1 + i * n_cell
        right_input_v = -1 + (i+1)*n_cell
        # 输出的range 的4个边界
        # left_low_1 = -1 * C
        x = (left_input_v+right_input_v)/2
        # x = left_input_v
        left_high = -C + k*x - B
        right_high = -C + k*x + B

        left_index = int((left_high +C) / m_cell + m/2)
        right_index = int((right_high +C) / m_cell+m/2)

        left_index_boundary = (left_index+0.5) * m_cell
        right_index_boundary = (right_index+0.5) * m_cell
        left_proportion = (left_high+2*C  - left_index_boundary) / m_cell
        right_proportion = (right_high+2*C  - right_index_boundary) / m_cell



        transform[left_index, i] = p_value * (1-left_proportion) + q_value * left_proportion
        transform[right_index, i] = p_value * right_proportion + q_value * (1-right_proportion)
        for j in range(left_index+1,right_index):
            transform[j,i] = p_value
        print(sum(transform[:,i]))
        sumALL = sum(transform[:,i])
        diff = sumALL - 1
        if diff > 0:
            # 表示要减少 left_index 和 right_index 的值
            temp111 = p_value - transform[left_index, i]
            if temp111 > diff:
                transform[left_index, i] += diff
            else:
                transform[left_index, i] = p_value
                transform[right_index, i] += diff-temp111
        elif diff < 0:
            if transform[right_index, i] -q_value > abs(diff):
                transform[right_index, i] += diff
            else:
                transform[right_index, i] = q_value
        # transform[i,left_index] = p_value * (1 - left_proportion) + q_value * left_proportion
        # transform[i,right_index] = p_value * right_proportion + q_value * (1-right_proportion)
        # for j in range(left_index+1,right_index):
        #     transform[i,j] = p_value
        # # print(sum(transform[i,j]))
        # sumALL = sum(transform[i,:])
        # diff = sumALL - 1
        # if diff > 0:
        #     # 表示要减少 left_index 和 right_index 的值
        #     temp111 = p_value - transform[i,left_index]
        #     if temp111 > diff:
        #         transform[i,left_index] += diff
        #     else:
        #         transform[i,left_index] = p_value
        #         transform[i ,right_index] += diff-temp111
        # elif diff < 0:
        #     if transform[i ,right_index] -q_value > abs(diff):
        #         transform[i ,right_index] += diff
        #     else:
        #         transform[i ,right_index] = q_value
    print( np.sum(transform, axis=1)) # 表示每一列每一列。
    max_iteration = 10000
    loglikelihood_threshold = 1e-3
    ns_hist, _ = np.histogram(noisy_samples, bins=randomized_bins, range=(-1*C, C))


    # return EM(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(ori_samples)
    return EMS(m, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(ori_samples)


def OPM_noise(samples,k,B,C,pr):
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
                perturbed_value =  temp * (2 * k) - C
            else:
                perturbed_value = rt + (temp - ppp) * (2 * k)

        hhh = 2*C / 100 -C
        if perturbed_value < hhh:
            count+=1
        res.append(perturbed_value)
    print(count)
    return np.array(res)

def EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold):
    # smoothing matrix
    smoothing_factor = 2
    # binomial_tmp = [binom(smoothing_factor, k) for k in range(smoothing_factor + 1)]
    binomial_tmp = [1,2,1] #[1,6,15,20,15,6,1] #[1, 2, 1]
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
    ori = np.random.normal(-0.3, 0.3, 50000)
    # ori = np.random.uniform(-1,1,100000)
    new_ori = []
    for i in ori:
        if i > 1 or i <-1:
            pass
        else:
            new_ori.append(i)
    # for i in ori2:
    #     new_ori.append(i)
    new_ori = np.array(new_ori)
    theta = OPM(new_ori,2,randomized_bins=256, domain_bins=256)
    print(theta)
    x = [i for i in range(len(theta))]
    plt.bar(x, theta)
    plt.xlabel(r'$\epsilon$', fontsize=16)
    plt.ylabel(u'2', fontsize=20)
    plt.tick_params(labelsize=20)
    # plt.text(1.19, 3.5, "key point", fontdict={'size': '16', 'color': 'b'})
    plt.title(u"Comparision and varying epsilon", fontsize=16)
    # plt.axvline(x=1.185, ls="-", c="red")  # 添加垂直直线
    # plt.axvline(x=1.29, ls="-", c="gray")  # 添加垂直直线
    plt.legend()
    plt.show()