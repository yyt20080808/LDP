#coding=utf-8
import numpy as np

import scipy
from numpy import linalg as LA
import matplotlib.pyplot as plt

def sw(ori_samples, l, h, eps, randomized_bins=1024, domain_bins=1024):
    ee = np.exp(eps)
    w = ((eps * ee) - ee + 1) / (2 * ee * (ee - 1 - eps)) * 2
    p = ee / (w * ee + 1)
    q = 1 / (w * ee + 1)

    samples = (ori_samples +1) / 2
    randoms = np.random.uniform(0, 1, len(samples))

    noisy_samples = np.zeros_like(samples)

    # report
    index = randoms <= (q * samples)
    noisy_samples[index] = randoms[index] / q - w / 2
    index = randoms > (q * samples)
    noisy_samples[index] = (randoms[index] - q * samples[index]) / p + samples[index] - w / 2
    index = randoms > q * samples + p * w
    noisy_samples[index] = (randoms[index] - q * samples[index] - p * w) / q + samples[index] + w / 2

    # report matrix
    m = randomized_bins
    n = domain_bins
    m_cell = (1 + w) / m
    n_cell = 1 / n

    transform = np.ones((m, n)) * q * m_cell
    for i in range(n):
        left_most_v = (i * n_cell)
        right_most_v = ((i + 1) * n_cell)

        ll_bound = int(left_most_v / m_cell)
        lr_bound = int((left_most_v + w) / m_cell)
        rl_bound = int(right_most_v / m_cell)
        rr_bound = int((right_most_v + w) / m_cell)

        ll_v = left_most_v - w / 2
        rl_v = right_most_v - w / 2
        l_p = ((ll_bound + 1) * m_cell - w / 2 - ll_v) * (p - q) + q * m_cell
        r_p = ((rl_bound + 1) * m_cell - w / 2 - rl_v) * (p - q) + q * m_cell
        if rl_bound > ll_bound:
            transform[ll_bound, i] = (l_p - q * m_cell) * (
            (ll_bound + 1) * m_cell - w / 2 - ll_v) / n_cell * 0.5 + q * m_cell
            transform[ll_bound + 1, i] = p * m_cell - (p * m_cell - r_p) * (
            rl_v - ((ll_bound + 1) * m_cell - w / 2)) / n_cell * 0.5
        else:
            transform[ll_bound, i] = (l_p + r_p) / 2
            transform[ll_bound + 1, i] = p * m_cell

        lr_v = left_most_v + w / 2
        rr_v = right_most_v + w / 2
        r_p = (rr_v - (rr_bound * m_cell - w / 2)) * (p - q) + q * m_cell
        l_p = (lr_v - (lr_bound * m_cell - w / 2)) * (p - q) + q * m_cell
        if rr_bound > lr_bound:
            if rr_bound < m:
                transform[rr_bound, i] = (r_p - q * m_cell) * (
                rr_v - (rr_bound * m_cell - w / 2)) / n_cell * 0.5 + q * m_cell

            transform[rr_bound - 1, i] = p * m_cell - (p * m_cell - l_p) * (
            (rr_bound * m_cell - w / 2) - lr_v) / n_cell * 0.5

        else:
            transform[rr_bound, i] = (l_p + r_p) / 2
            transform[rr_bound - 1, i] = p * m_cell

        if rr_bound - 1 > ll_bound + 2:
            transform[ll_bound + 2: rr_bound - 1, i] = p * m_cell
        # print(sum(transform[:, i]))

    max_iteration = 10000
    loglikelihood_threshold = 1e-3
    ns_hist, _ = np.histogram(noisy_samples, bins=randomized_bins, range=(-w / 2, 1 + w / 2))
    # return EM(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(ori_samples)
    return EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(ori_samples)


def EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold):
    # smoothing matrix
    smoothing_factor = 2
    binomial_tmp = [1,2,1]
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
            # print("stop when", imporve / old_logliklihood, loglikelihood_threshold)
            break

        old_logliklihood = logliklihood

        r += 1
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
    # ori = np.random.normal(0.3, 0.5, 200000 )
    binnumber = 200
    ori = np.random.uniform(0, 1, 500000)
    new_ori = []
    for i in ori:
        if i > 1 or i <-1:
            pass
        else:
            new_ori.append(i)
    new_ori = np.array(new_ori)
    theta = sw(new_ori,0,1,1,randomized_bins=binnumber, domain_bins=binnumber)
    print(theta)
    x = [i for i in range(len(theta))]
    plt.bar(x, theta)
    plt.xlabel(r'$\epsilon$', fontsize=16)
    plt.ylabel(u'MAE (log)', fontsize=20)
    plt.tick_params(labelsize=20)
    # plt.text(1.19, 3.5, "key point", fontdict={'size': '16', 'color': 'b'})
    # plt.title(u"Comparision and varying epsilon", fontsize=16)
    # plt.axvline(x=1.185, ls="-", c="red")  # 添加垂直直线
    # plt.axvline(x=1.29, ls="-", c="gray")  # 添加垂直直线
    plt.legend()
    plt.show()