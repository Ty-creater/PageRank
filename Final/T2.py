import numpy as np
import scipy as sp
import scipy.sparse
import sys
import time
import gc


def T2(graph_path, output_path, sampling_ratio):
    node_set = set()
    with open(graph_path, 'r') as file:
        for line in file:
            if line.startswith('%') or line.startswith('#'):
                continue
            data = line.strip().split()
            from_node = int(data[0])
            to_node = int(data[1])
            node_set.add(from_node)
            node_set.add(to_node)
    node_dict = {element: index for index, element in enumerate(node_set)}
    node_dict_t = {index: element for index, element in enumerate(node_set)}
    print("collect node_set")
    sys.stdout.flush()
    del node_set
    gc.collect()

    # 加载数据
    row, col = [], []
    with open(graph_path, 'r') as file:
        for line in file:
            if line.startswith('%') or line.startswith('#'):
                continue
            data = line.strip().split()
            from_node = int(data[0])
            to_node = int(data[1])
            row.append(node_dict[from_node])
            col.append(node_dict[to_node])
            row.append(node_dict[to_node])
            col.append(node_dict[from_node])
    data = np.ones(len(row))

    # 生成转移矩阵A
    n = len(node_dict_t)
    A = sp.sparse.coo_array((data, (row, col)), shape=(n, n), dtype=float)
    S = A.sum(axis=1)
    S[S != 0] = 1.0 / S[S != 0]
    Q = sp.sparse.csr_array(sp.sparse.spdiags(S.T, 0, *A.shape))
    A = Q @ A

    print("collect node_dict")
    sys.stdout.flush()
    del node_dict
    gc.collect()

    # 原图的边数
    original_edges_num = A.nnz

    # TODO 不同分布抽点
    # 计算p分布（使用行和列范数）
    col_sums = A.power(2).sum(axis=0)
    row_sums = A.power(2).sum(axis=1)
    p = np.multiply(col_sums, row_sums)
    probabilities = p / sum(p)

    # 列范数抽
    # total_sums = A.power(2).sum()
    # col_sums = A.power(2).sum(axis=0)
    # col_distribution = col_sums / total_sums

    # 行范数抽
    # total_sums = A.power(2).sum()
    # row_sums = A.power(2).sum(axis=1)
    # row_distribution = row_sums / total_sums

    start_time = time.time()  # 开始计时

    c = int(len(node_dict_t) * sampling_ratio)
    # 使用行和列范数分布抽
    sampled_index = np.random.choice(A.shape[1], size=c, replace=False, p=probabilities)
    # # 列范数分布抽
    # sampled_index = np.random.choice(A.shape[1], size=c, replace=False, p=col_distribution)
    # # 行范数分布抽
    # sampled_index = np.random.choice(A.shape[1], size=c, replace=False, p=row_distribution)
    # # 均匀分布抽
    # sampled_index = np.random.choice(A.shape[1], size=c, replace=False, p=None)
    sampled_index.sort()
    C = A[:, sampled_index]  # 列向量，n*c
    C = sp.sparse.csc_array(C)
    R = A[sampled_index, :]  # 行向量，c*n

    print("the number of C edges: ", C.nnz / original_edges_num)
    sys.stdout.flush()

    ## TODO 如果要做Scaling的话，需要以下代码
    # probabilities = np.sqrt(probabilities * c)
    # # 行遍历
    # for i in range(R.shape[0]):
    #     start_idx = R.indptr[i]
    #     end_idx = R.indptr[i + 1]
    #     # 遍历当前行的非零元素
    #     for j in range(start_idx, end_idx):
    #         col_idx = R.indices[j]
    #         value = R.data[j]
    #         R[i, col_idx] = value / probabilities[sampled_index[i]]
    #
    # # 列遍历
    # for j in range(C.shape[1]):
    #     start_idx = C.indptr[j]
    #     end_idx = C.indptr[j + 1]
    #     for i in range(start_idx, end_idx):
    #         row_idx = C.indices[i]
    #         value = C.data[i]
    #         C[row_idx, j] = value / probabilities[sampled_index[j]]

    print("start compute pagerank")
    sys.stdout.flush()

    # 开始迭代计算PageRank
    r = np.repeat(1.0 / n, n)
    P = np.repeat(1.0 / n, n)
    c_v = np.repeat(1.0 / c, c)
    alpha = 0.85
    r = R @ r
    tol = 1 / n / 10
    for _ in range(sys.maxsize):
        r_last = r
        r = C @ r
        r = R @ r
        r = (1 - pow(alpha, 2)) * r + pow(alpha, 2) * c_v
        # 计算误差
        err = np.absolute(r - r_last).sum()
        if err < c * tol:
            r = C @ r
            r = (alpha / (1 + alpha)) * r + (1 - alpha) * P
            r = r / np.linalg.norm(r, ord=1)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"execution_time：{execution_time:.2f} s")
            print("the number of iterations: ", _ + 1)
            # 保存结果
            with open(output_path, 'w+') as file:
                for i in range(len(node_dict_t)):
                    file.write(f"{node_dict_t[i]}\t{r[i]:.17f}\n")
                break


args = sys.argv
graph_path = args[1]
output_path = args[2]
sampling_ratio = float(args[3])
T2(graph_path, output_path, sampling_ratio)

