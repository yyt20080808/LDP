
import numpy as np
from scipy.stats import laplace
import math
# 定义Laplace分布的数目
num_laplace = 200 # set the number of laplace distribution
n = 3000  # number of noised reports
from scipy.optimize import minimize
np.random.seed(123)
def generate_data():
    res = []
    a = np.random.normal(0.5, math.sqrt(0.02), 1000)
    res.extend(a)
    b = np.random.normal(0.4, math.sqrt(0.01), 1000)
    res.extend(b)
    c = np.random.normal(-0.1, math.sqrt(0.002), 1000)
    res.extend(c)
    new_res = []
    for i in res:
        if i <= 1 and i >= -1:
            new_res.append(i)
    print("mean:",sum(new_res)/len(new_res))
    return new_res


def laplaceMechanism(a,epsilon_list):
    res_directsum = []
    res_EM = []
    for epsilon in epsilon_list:
        accuracy = 0
        accuracy_EM = 0
        for times in range(5):
            print(epsilon,times)
            res=0
            all = sum(a)
            noised_reports = []
            for j in a:
                noise = np.random.laplace(0, 2 / epsilon, 1)
                res += (j + noise[0])
                noised_reports.append(j+noise[0])
            length = len(a)
            accuracy += (all / length - res / length)**2
            result = minimize(
                lambda x: -log_likelihood(epsilon,x,noised_reports),
                x0=np.ones(num_laplace) / num_laplace,
                method='SLSQP',
                bounds=[(0, 1) for i in range(num_laplace)],
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            )
            Em_est = calmean(result.x)
            accuracy_EM += (Em_est - res / length)**2
            # 开始 EM
        res_directsum.append(accuracy / 5)
        res_EM.append(accuracy_EM/5)
    return res_directsum,res_EM

# 生成样本数据

# 定义混合模型的似然函数
def log_likelihood(epsilon,weights,data):
    # weights: 每个Laplace分布的权重
    means = np.linspace(-1, 1, num_laplace)
    if np.any(weights < 0) or np.any(weights > 1):
        return -np.inf
    else:
        mixture = np.sum([
            weights[i] * laplace.logpdf(data, loc=means[i], scale=2/epsilon)
            for i in range(num_laplace)
        ], axis=0)
        return np.sum(mixture)


def calmean(res):
    normalized_arr = res / np.sum(res)
    mean_est = 0
    for i in range(num_laplace):
        mean_est += normalized_arr[i] * (i + 0.5)
    sum_OPMnos = (mean_est / num_laplace - 0.5) * 2  # 转换到 -1 到 1 的区间
    return sum_OPMnos


if __name__ == "__main__":
    a =generate_data()
    ep_list = [0.5,0.75,1]
    res_direct, res_EM= laplaceMechanism(a,ep_list)
    print("样本均值结果:")
    print(res_direct)
    print("混合laplace分布计算结果:")
    print(res_EM)