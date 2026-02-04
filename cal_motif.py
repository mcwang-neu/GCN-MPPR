# -*-coding: utf-8 -*-
from numpy import *
# from motif_construct_direct import *
from motif_construct_direct import *
from scipy.sparse import diags
import scipy.sparse as sp
import pickle
from typing import Union
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sparsegraph import SparseGraph
MAX_TIME = 30000
data_dir = Path(__file__).parent

def graphMove(a):
    # 构造转移矩阵, we will construct the transfer matrix
    b = transpose(a)
    # b为a的转置矩阵
    c = zeros((a.shape), dtype=float)
    for j in range(a.shape[1]):
        if j % 1000 == 0:
            pass
        if b[j].sum() == 0:
            continue
        else:
            c[j] = b[j] / (b[j].sum())
    c = transpose(c)
    return c


def graphMove_new(a):
    for number in range(0, np.size(a, 0)):
        if number % 10000 == 0:
            pass
        # row_sum = np.sum(a[number, :].toarray())
        row_sum = a.getrow(number).sum()
        if row_sum == 0:
            continue
        else:
            for number_col in a.indices[a.indptr[number]:a.indptr[number+1]]:
                a[number, number_col] = a[number, number_col] / row_sum
    a = transpose(a)
    return a


def graphMove_newest(a):
    d = a.sum(0)
    d = np.array(d)
    d = d[0]
    dd = map(lambda x: 0 if x==0 else np.power(x, -0.5), d)
    D_matrix =diags(dd, 0)
    C = D_matrix.dot(a)
    C = C.dot(D_matrix)
    a = C
    a = transpose(a)
    return a


def firstPr(c):
    # pr值得初始化
    pr = zeros((c.shape[0], 1), dtype=float)
    # 构造一个存放pr值得矩阵
    for i in range(c.shape[0]):
        pr[i] = float(1) / c.shape[0]
    return pr


def pageRank(p, m, v):
    e = np.ones((m.shape[0], 1))
    n = m.shape[0]
    count = 0
    # m_spare = csr_matrix(m)
    # we will use the equation to compute the value of pagerank 计算pageRank值
    while count <= MAX_TIME:
        # 判断pr矩阵是否收敛,(v == p*dot(m,v) + (1-p)*v).all()判断前后的pr矩阵是否相等，若相等则停止循环
        if count % 10000 == 0:
            pass
        v = p * m.dot(v) + ((1 - p) / n) * e
        count = count + 1
    return v


def test():
    types = ['M1', 'M2', 'M3','M4', 'M5', 'M6','M7'] # , 'M8', 'M9','M10', 'M11', 'M12', 'M13', 'M14'
    for type in types:
        print(type)
        a, entry_unique = construct_motif('../data/Ciao/trust_network_ciao.txt', 1, type,
                                          0.08)  # '../../data/Ciao/trust_network_ciao.txt'
        M = graphMove_new(a)

        def calc_ppr_exact(adj_matrix: sp.spmatrix, alpha: float) -> np.ndarray:
            def calc_A_hat(adj_matrix):  # -> Union[sp.spmatrix, sp.sparray]
                nnodes = adj_matrix.shape[0]
                A = adj_matrix + sp.eye(nnodes)
                D_vec = np.sum(A, axis=1)
                if isinstance(A, sp.spmatrix):
                    D_vec = D_vec.A1
                D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
                D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
                return D_invsqrt_corr @ A @ D_invsqrt_corr
            nnodes = adj_matrix.shape[0]
            M = calc_A_hat(adj_matrix)
            A_inner = sp.eye(nnodes) - (1 - alpha) * M
            return alpha * np.linalg.inv(A_inner.toarray())

        mppr = calc_ppr_exact(M, 0.1)

        with open('mppr_ciao/' + type + '.pkl', 'wb') as f:
            pickle.dump(mppr, f)

        # print(type(M))
        # pr = firstPr(M)
        # p = 0.8  # 引入浏览当前网页的概率为p,假设p=0.8
        # v = pageRank(p, M, pr)  # 计算pr值
        # result_temp = {}
        # for number in range(len(v)):
        #     result_temp[entry_unique[number]] = v[number]
        # dict = sorted(result_temp.items(), key=lambda item: item[0], reverse=False)
        # output = open('result_Ciao_M5_alpha0.08.txt', 'w')
        # for i in range(len(dict)):
        #     a = "%d;%lf\n" % (dict[i][0], dict[i][1])
        #     output.write(a)
        # output.close()


def cal_main(graph_name):
    def load_dataset(name: str, directory: Union[Path, str] = data_dir):
        def load_from_npz(file_name: str) -> SparseGraph:
            with np.load(file_name, allow_pickle=True) as loader:
                loader = dict(loader)
                dataset = SparseGraph.from_flat_dict(loader)
            return dataset
        if isinstance(directory, str):
            directory = Path(directory)
        if not name.endswith('.npz'):
            name += '.npz'
        path_to_file = directory / name
        if path_to_file.exists():
            return load_from_npz(path_to_file)
        else:
            raise ValueError("{} doesn't exist.".format(path_to_file))
    graph_name = (graph_name) # amazon_electronics_photo
    graph = load_dataset('data/' + graph_name)
    graph.standardize(select_lcc=True)
    print(graph)
    a = graph.adj_matrix.A
    print(type(graph))

    output = open('data/' + graph_name + '.txt', 'w')
    for i in range(len(a)):
        for j in range(len(a)):
            if a[i, j] == 1:
                tt = "%d;%d\n" % (i, j)
                output.write(tt)
    output.close()

    for motif_type in ['M1', 'M2', 'M3','M4', 'M5', 'M6', 'M7']:
        print(motif_type)
        alpha_value = 0.08

        a, entry_unique = construct_motif('data/' + graph_name + '.txt', 1, motif_type, alpha_value)
        M = graphMove_new(a)
        #print(M.shape)
        #print(type(M))
        def calc_ppr_exact(adj_matrix: sp.spmatrix, alpha: float) -> np.ndarray:
            def calc_A_hat(adj_matrix):  # -> Union[sp.spmatrix, sp.sparray]
                nnodes = adj_matrix.shape[0]
                A = adj_matrix + sp.eye(nnodes)
                D_vec = np.sum(A, axis=1)
                if isinstance(A, sp.spmatrix):
                    D_vec = D_vec.A1
                D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
                D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
                return D_invsqrt_corr @ A @ D_invsqrt_corr

            nnodes = adj_matrix.shape[0]
            M = calc_A_hat(adj_matrix)
            A_inner = sp.eye(nnodes) - (1 - alpha) * M
            return alpha * np.linalg.inv(A_inner.toarray())

        ap = 0.1
        mppr = calc_ppr_exact(M, ap)

        import os

        path = "result/result_" + graph_name

        if not os.path.exists(path):
            os.makedirs(path)
            # print("Folder created")

        with open('result/result_' + graph_name + '/' + motif_type + '.pkl', 'wb') as f:
            pickle.dump(mppr, f)
