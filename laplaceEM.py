import numpy as np
from scipy.stats import laplace
import math
# Generate some sample data from a Laplace mixture model
n = 3000
num_laplace = 21
means =  np.linspace(0, 20, num_laplace)
# proportions = np.array([0.2, 0.3, 0.5])
# z = np.random.choice(len(loc), size=n, p=proportions)
# data = np.concatenate([
#     laplace.rvs(loc=mean, scale=2/epsilon, size=int(n/num_laplace))
#     for mean in means
# ])

def generate_data():
    res = []
    a = np.random.normal(5, math.sqrt(0.02), 1000)
    # a = np.random.uniform(-1, 1, 100)
    res.extend(a)
    b = np.random.normal(4, math.sqrt(0.01), 1000)
    res.extend(b)
    c = np.random.normal(9, math.sqrt(0.2), 1000)
    res.extend(c)
    print("mean:",sum(res)/len(res))
    return res

def laplaceMechanism(a,epsilon_list):
    exper = 10
    res_directsum = []
    res_EM = []
    for epsilon in epsilon_list:
        accuracy = 0
        accuracy_EM = 0
        for times in range(exper):
            all = sum(a)
            print(epsilon,times,all)
            res = 0
            noised_reports = []
            for j in a:
                noise = np.random.laplace(0, 1/epsilon, 1)
                res += (j + noise[0])
                noised_reports.append(j+noise[0])
            length = len(a)
            accuracy += (all / length - res / length)**2

            noised_reports = np.array(noised_reports)
            result = EM(noised_reports,epsilon)
            print(result)
            Em_est = calmean(result)
            print(res / length, Em_est)
            accuracy_EM += (Em_est - all / length)**2
            # 开始 EM
        res_directsum.append(accuracy / exper)
        res_EM.append(accuracy_EM/exper)
    return res_directsum,res_EM

def EM(data,eps):
    weights = np.ones(num_laplace) / num_laplace
    # Run EM algorithm
    tolerance = 1e-6
    max_iter = 1000
    ll_old = -np.inf
    for iter in range(max_iter):
        # E-step: compute expected values of latent variable z
        probs = np.zeros((n, num_laplace))
        for j in range(num_laplace):
            probs[:, j] = weights[j] * np.exp(-np.abs(data - means[j]) / (1/eps))
        probs /= probs.sum(axis=1, keepdims=True)

        # M-step: update mixture weights
        weights = probs.mean(axis=0)

        # Compute log-likelihood and check for convergence
        ll = np.sum(np.log(probs @ weights))
        if ll - ll_old < tolerance:
            break
        ll_old = ll
    return weights

def calmean(res):
    print(res)
    normalized_arr = res / np.sum(res)
    mean_est = 0
    for i in range(num_laplace):
        mean_est += normalized_arr[i] * (i)
    # mean_est = (mean_est / num_laplace - 0.5) * 2  # 转换到 -1 到 1 的区间
    return mean_est

if __name__ == "__main__":
    a =generate_data()
    ep_list = [1]
    res_direct, res_EM = laplaceMechanism(a, ep_list)
# Print estimated mixture weights
    print("样本均值结果:")
    print(res_direct)
    print("混合laplace分布计算结果:")
    print(res_EM)
