# @author lsg
# @date 2023/3/22
# @file two.py
import math

import numpy as np
from numpy.dual import cholesky, inv
from scipy.sparse import spdiags


def airPLS(X, lam, order, wep=0.05, p=0.05, itermax=20):
    (m,n) = X.shape
    Z = np.empty([m,n])
    D = np.diff(np.eye(n), order, axis=0)
    DD = np.matmul(lam * D.T,D)
    for i in range(m):
        w = np.ones([n,1]).T
        x = X[i,:]
        for j in range(1, itermax+1):
            W = spdiags(w, 0, n, n)
            C = cholesky(W + DD)
            z = np.matmul(inv(C), np.matmul(inv(C.T), (w * x).T)).T
            d = x - z
            dssn = np.abs(sum(d[d<0]))
            if dssn < 0.001 * sum(np.abs(x)):
                break
            w[d>=0] = 0
            w[0][:math.ceil(n*wep)] = p
            w[0][n-math.floor(n*wep)-1:] = p
            to_exp = abs(d[d<0])/dssn
            w[d<0] = j * np.exp(to_exp)
        Z[i,:] = z
    Xc=X-Z
    return Xc,Z


if __name__ == '__main__':
    # 构造示例数据，shape为（10,100）
    data = np.random.rand(10, 100)

    # 调用函数去除基线漂移
    Xc, Z = airPLS(data, 10 ** 6, 1)

    # 输出去除基线漂移后的多元谱数据
    print("去除基线漂移后的多元谱数据：\n", Xc)

    # 输出基线漂移
    print("基线漂移：\n", Z)











