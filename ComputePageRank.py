import math

import networkx as nx
import networkit as nk
import random
import sys
import time
from networkit.centrality import PageRank
import gc
from concurrent.futures import ThreadPoolExecutor

# def write_to_file(node_pagerank):
#     node, pagerank = node_pagerank
#     return f"{node}\t{pagerank:.17f}\n"

# 使用networkx计算相对准确的pagerank值，用于做ground truth
def computeGlobalPageRank(graph_Path):
    G = nx.Graph()
    with open(graph_Path, 'r') as file:
        for line in file:
            if line.startswith('%'): continue
            data = line.strip().split()
            from_node = int(data[0])
            to_node = int(data[1])
            G.add_edge(from_node, to_node)

    pagerank_list = nx.pagerank(G, tol=1e-17, max_iter=100000000, alpha=0.85)
    sorted_pagerank = sorted(pagerank_list.items(), key=lambda x: x[0])

    with open('PrecisePageRankResult/contiguous/pageRank.txt', 'w+') as file:
        for node, pagerank in sorted_pagerank:
            file.write(f"{node}\t{pagerank:.17f}\n")


# 使用Networkit来计算原图的PageRank
def computeGlobalPageRankWithNetworkit(graph_path, output_path):
    node_set = set()
    original_edges = []
    with open(graph_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('%') or line.startswith('#'):
                continue
            data = line.strip().split()
            from_node = int(data[0])
            to_node = int(data[1])
            original_edges.append((from_node, to_node))
            node_set.add(from_node)
            node_set.add(to_node)

    del lines
    gc.collect()

    node_dict = {element: index for index, element in enumerate(node_set)}
    node_dict_t = {index: element for index, element in enumerate(node_set)}

    G = nk.Graph(directed=True, n=len(node_dict))
    for from_node, to_node in original_edges:
        from_node_index = node_dict[from_node]
        to_node_index = node_dict[to_node]
        G.addEdge(from_node_index, to_node_index)
        G.addEdge(to_node_index, from_node_index)

    pr = nk.centrality.PageRank(G, tol=1 / len(node_dict) / 100, damp=0.85)
    pr.run()
    pagerank_scores = pr.scores()

    with open(output_path, 'w+') as file:
        for node, pagerank in enumerate(pagerank_scores):
            file.write(f"{node_dict_t[node]}\t{pagerank:.17f}\n")


def savePageRankList(pagerank_list, output_path):
    with open(output_path, 'w+') as file:
        for node, pagerank in pagerank_list.items():
            file.write(f"{node}\t{pagerank:.17f}\n")


def computeSubPageRankByTheta(graph_path, output_path, theta):
    G = nx.Graph()
    count = 0
    with open(graph_path, 'r') as file:
        for line in file:
            if line.startswith('%') or line.startswith('#'): continue
            data = line.strip().split()
            from_node = int(data[0])
            to_node = int(data[1])
            random_number = random.uniform(0, 1)
            if random_number > theta:
                G.add_edge(from_node, to_node)
            else:
                count += 1

    print(f"the number of the deleted edges: {count}")
    print(f"After Edge Sparsification, the number of nodes: {len(G.nodes)}")
    pagerank_list = nx.pagerank(G, tol=1e-6, max_iter=10000000000000, alpha=0.85)
    savePageRankList(pagerank_list, output_path)


def computeSubPageRankByThetaWithNetworkit(graph_path, output_path, theta):
    # node_set = set()
    # sampled_edges = []
    # sampled_edges_num = 0
    # edges_total = 0
    # with open(graph_path, 'r') as file:
    #     for line in file:
    #         if line.startswith('%') or line.startswith('#'):
    #             continue
    #         edges_total += 1
    #         data = line.strip().split()
    #         from_node = int(data[0])
    #         to_node = int(data[1])
    #         random_number = random.uniform(0, 1)
    #         if random_number < theta:
    #             sampled_edges_num += 1
    #             node_set.add(from_node)
    #             node_set.add(to_node)
    #             sampled_edges.append((from_node, to_node))
    # print("the number of selected_sub_graph edges: ", sampled_edges_num / edges_total)
    # print(f"After Edge Sparsification, the number of nodes: {len(node_set)}")
    # node_dict = {element: index for index, element in enumerate(node_set)}
    # node_dict_t = {index: element for index, element in enumerate(node_set)}

    node_set = set()
    original_edges = []
    with open(graph_path, 'r') as file:
        for line in file:
            if line.startswith('%') or line.startswith('#'):
                continue
            data = line.strip().split()
            from_node = int(data[0])
            to_node = int(data[1])
            original_edges.append((from_node, to_node))
            node_set.add(from_node)
            node_set.add(to_node)

    node_dict = {element: index for index, element in enumerate(node_set)}  # 节点id:原图index
    node_dict_t = {index: element for index, element in enumerate(node_set)}  # 原图index:节点id

    print("start construct graph")
    sys.stdout.flush()

    del node_set
    gc.collect()

    start_time = time.time()

    # 原始图
    G = nk.Graph(directed=True, n=len(node_dict))
    for from_node, to_node in original_edges:
        random_number = random.uniform(0, 1)
        if random_number < theta:
            from_node_index = node_dict[from_node]
            to_node_index = node_dict[to_node]
            G.addEdge(from_node_index, to_node_index)
            G.addEdge(to_node_index, from_node_index)

    print("the number of selected_sub_graph edges: ", G.numberOfEdges() / len(original_edges) / 2)
    sys.stdout.flush()

    del original_edges
    gc.collect()

    sub_nodes = set()
    for node in G.iterNodes():
        if G.degreeIn(node) != 0 or G.degreeOut(node) != 0:
            sub_nodes.add(node)
    print(f"After Edge Sparsification, the number of nodes: {len(sub_nodes)}")
    sys.stdout.flush()

    # 计算pagerank
    pr = nk.centrality.PageRank(G, tol=1e-6, damp=0.85, distributeSinks=True)
    pr.maxIterations = 100
    pr.run()
    pagerank_scores = pr.scores()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"execution_time：{execution_time:.2f} s")
    print("the number of iterations: ", pr.numberOfIterations())
    sys.stdout.flush()

    # data_to_write = [(node_dict_t[node], pagerank) for node, pagerank in enumerate(pagerank_scores) if node in sub_nodes]
    # with open(output_path, 'w+') as file, ThreadPoolExecutor() as executor:
    #     file.writelines(executor.map(write_to_file, data_to_write))

    with open(output_path, 'w+') as file:
        for node, pagerank in enumerate(pagerank_scores):
            if node in sub_nodes:
                file.write(f"{node_dict_t[node]}\t{pagerank:.17f}\n")


def computeSubPageRankByThetaWithNetworkit2(graph_path, output_path, theta):
    G = nk.Graph(directed=True)
    count = 0
    original_edges = []
    with open(graph_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('%') or line.startswith('#'):
                continue
            data = line.strip().split()
            from_node = int(data[0])
            to_node = int(data[1])
            original_edges.append((from_node, to_node))
    del lines
    gc.collect()

    start_time = time.time()
    for from_node, to_node in original_edges:
        random_number = random.uniform(0, 1)
        if random_number < theta:
            count += 1
            G.addEdge(from_node - 1, to_node - 1, addMissing=True)
            G.addEdge(to_node - 1, from_node - 1)

    print("the number of selected_sub_graph edges: ", count / len(original_edges))
    del original_edges
    gc.collect()
    sub_nodes = set()
    for node in G.iterNodes():
        if G.degreeIn(node) != 0 or G.degreeOut(node) != 0:
            sub_nodes.add(node)
    print(f"After Edge Sparsification, the number of nodes: {len(sub_nodes)}")
    sys.stdout.flush()

    pr = nk.centrality.PageRank(G, tol=1/G.numberOfNodes()/10, damp=0.85)
    pr.run()
    pagerank_scores = pr.scores()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"execution_time：{execution_time:.2f} s")
    print("the number of iterations: ", pr.numberOfIterations())

    with open(output_path, 'w+') as file:
        for node, pagerank in enumerate(pagerank_scores):
            if node in sub_nodes:
                file.write(f"{node+1}\t{pagerank:.17f}\n")


# 使用NetworkX，有权图按权值删边
def computeSubPageRankByTau(graph_path, output_path, tau):
    G = nx.DiGraph()
    with open(graph_path, 'r') as file:
        for line in file:
            if line.startswith('%') or line.startswith('#'): continue
            data = line.strip().split()
            from_node = int(data[0])
            to_node = int(data[1])
            weight = float(data[2])
            G.add_edge(from_node, to_node, weight=weight)
            G.add_edge(to_node, from_node, weight=weight)

    for node in G.nodes:
        total_weight = 0
        for neighbor in G.neighbors(node):
            total_weight += G[node][neighbor]['weight']
        for neighbor in G.neighbors(node):
            G[node][neighbor]['weight'] /= total_weight

    edges_to_remove = []
    print(f"the number of the deleted edges: {len(edges_to_remove)}")
    for edge in G.edges(data=True):
        source, target, data = edge
        weight = data.get('weight', 1.0)
        if weight < tau:
            edges_to_remove.append((source, target))

    for edge in edges_to_remove:
        G.remove_edge(*edge)

    nodes_to_remove = []
    for node in G.nodes:
        if G.out_degree(node) == 0 and G.in_degree(node) == 0:
            nodes_to_remove.append(node)

    for node in nodes_to_remove:
        G.remove_node(node)

    print(f"After Edge Sparsification, the number of nodes: {len(G.nodes)}")
    pagerank_list = nx.pagerank(G, tol=1e-6, max_iter=10000000000000, alpha=0.85)
    savePageRankList(pagerank_list, output_path)


# 使用Networkit，有权图按权值删边
def computeSubPageRankByTauWithNetworkit(graph_path, output_path, tau):
    G = nk.Graph(directed=True, weighted=True)
    min_weight = 0
    max_weight = 0
    with open(graph_path, 'r') as file:
        for line in file:
            if line.startswith('%') or line.startswith('#'): continue
            data = line.strip().split()
            from_node = int(data[0])
            to_node = int(data[1])
            weight = float(data[2])
            if weight > max_weight:
                max_weight = weight
            elif weight < min_weight:
                min_weight = weight
            G.addEdge(from_node-1, to_node-1, weight, addMissing=True)
            G.addEdge(to_node-1, from_node-1, weight)

    # 原图的边数
    original_edges_num = G.numberOfEdges()

    # 仅限于Wikipedia conflict这样的weight有负数的数据集
    G2 = nk.Graph(directed=True, weighted=True)
    for edge in G.iterEdgesWeights():
        from_node, to_node, weight = edge
        new_weight = (weight - min_weight) / (max_weight - min_weight)
        G2.addEdge(from_node, to_node, new_weight, addMissing=True)

    # 行归一化
    out_neighbor_weight_sum = {}
    for node in G2.iterNodes():
        out_neighbor_weight_sum[node] = G2.weightedDegree(node)

    # 删边
    G3 = nk.Graph(directed=True, weighted=True)
    count = 0
    for edge in G2.iterEdgesWeights():
        from_node, to_node, weight = edge
        new_weight = weight / out_neighbor_weight_sum[from_node]
        if new_weight < tau:
            count += 1
        else:
            G3.addEdge(from_node, to_node, new_weight, addMissing=True)
    print(f"the number of the deleted edges: {count}")
    print(f"the proportion of deleting edges : {count/original_edges_num}")
    print(f"After Edge Sparsification, the number of nodes: {G3.numberOfNodes()}")

    # 计算pagerank
    pr = nk.centrality.PageRank(G3, tol=1e-6, damp=0.85)
    pr.run()
    pagerank_scores = pr.scores()
    print("the number of iterations: ", pr.numberOfIterations())

    # 保存结果
    with open(output_path, 'w+') as file:
        for node, pagerank in enumerate(pagerank_scores):
            file.write(f"{node + 1}\t{pagerank:.17f}\n")


# 以矩阵的方式来实现ApproxRank
def approxRankWithNetworkit(graph_path, output_path, node_num, sampling_ratio):
    import numpy as np
    import scipy as sp
    import scipy.sparse

    # 因为存在一些数据集的id不是自增的，所以节点id的最大值可能不是节点总数
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

    print("node_set_len ", len(node_set))
    print("node_dict_len ", len(node_dict))
    print("node_dict_t_len ", len(node_dict_t))
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
    n = node_num
    A = sp.sparse.coo_array((data, (row, col)), shape=(n, n), dtype=float)
    S = A.sum(axis=1)
    S[S != 0] = 1.0 / S[S != 0]
    Q = sp.sparse.csr_array(sp.sparse.spdiags(S.T, 0, *A.shape))
    A = Q @ A

    print("collect node_dict")
    sys.stdout.flush()
    del node_dict
    gc.collect()

    start_time = time.time()

    print("start construct new matrix")
    sys.stdout.flush()

    # TODO：转移矩阵左上角
    sub_graph_num = int(n * sampling_ratio)
    selected_node_indices = np.random.choice(np.arange(0, n), size=sub_graph_num, replace=False)
    upper_left = A[selected_node_indices][:, selected_node_indices]

    # TODO：转移矩阵左下角
    sum_vector = np.array(A.sum(axis=0)) - np.array(A[selected_node_indices, :].sum(axis=0))
    lower_left_selected = sum_vector[selected_node_indices]
    lower_left = lower_left_selected / (n - len(selected_node_indices))

    # TODO：转移矩阵右上角和转移矩阵右下角
    vstack_result = sp.sparse.vstack([upper_left, sp.sparse.csr_matrix(lower_left)])
    right_column = 1 - np.array(vstack_result.sum(axis=1)).flatten()
    final_matrix = sp.sparse.hstack([vstack_result, sp.sparse.csr_matrix(right_column).T])
    print("the number of subgraph's edges: ", final_matrix.nnz / A.nnz)
    sys.stdout.flush()

    # 初始向量
    # R = np.repeat(1.0 / (sub_graph_num+1), sub_graph_num+1)
    R_1 = np.repeat(1.0 / n, sub_graph_num)
    R_2 = np.array([(n - sub_graph_num) / n])
    R = np.concatenate((R_1, R_2))

    # 跳转概率分布
    P_1 = np.repeat(1.0 / n, sub_graph_num)
    P_2 = np.array([(n - sub_graph_num) / n])
    P = np.concatenate((P_1, P_2))

    print("start compute pagerank")
    sys.stdout.flush()

    # PageRank迭代
    alpha = 0.85
    tol = 1 / node_num / 10
    for _ in range(sys.maxsize):
        R_last = R
        R = alpha * (R @ final_matrix) + (1 - alpha) * P
        # 计算误差
        err = np.absolute(R - R_last).sum()
        if err < (sub_graph_num + 1) * tol:
            # 保存结果
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"execution_time：{execution_time:.2f} s")
            print("the number of iterations: ", _ + 1)
            with open(output_path, 'w+') as file:
                for i in range(0, len(selected_node_indices)):
                    file.write(f"{node_dict_t[selected_node_indices[i]]}\t{R[i]:.17f}\n")
                break


# 元素采样，矩阵
def elementSampling(graph_path, output_path, alpha, theta):
    import numpy as np
    import scipy as sp
    import scipy.sparse
    from scipy.sparse.linalg import norm

    # 因为存在一些数据集的id不是自增的，所以节点id的最大值可能不是节点总数
    node_set = set()
    with open(graph_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('%') or line.startswith('#'):
                continue
            data = line.strip().split()
            from_node = int(data[0])
            to_node = int(data[1])
            node_set.add(from_node)
            node_set.add(to_node)

    del lines
    gc.collect()

    node_dict = {element: index for index, element in enumerate(node_set)}
    node_dict_t = {index: element for index, element in enumerate(node_set)}

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
    n = len(node_dict)
    A = sp.sparse.coo_array((data, (row, col)), shape=(n, n), dtype=float)
    S = A.sum(axis=1)
    S[S != 0] = 1.0 / S[S != 0]
    Q = sp.sparse.csr_array(sp.sparse.spdiags(S.T, 0, *A.shape))
    A = Q @ A

    start_time = time.time()

    frobenius_norm = norm(A, ord='fro')
    original_edges_nums = A.nnz
    s = original_edges_nums / alpha
    delta = theta * frobenius_norm / math.sqrt(s)
    deltaC = 1 / delta

    print(f"f_norm: {frobenius_norm}")
    print(f"s: {s}")
    print(f"delta: {delta}")
    print(f"deltaC: {deltaC}")
    sys.stdout.flush()

    # 获取非零元素的行和列索引
    # nonzero_indices = A.nonzero()
    sub_node_set = set()
    # for i, j in zip(nonzero_indices[0], nonzero_indices[1]):
    #     if A[i, j] > delta:
    #         sub_node_set.add(i)
    #         sub_node_set.add(j)
    #         A[i, j] = max(A[i, j], pow(frobenius_norm, 2) / (s * A[i, j]))
    #     else:
    #         random_number = random.uniform(0, 1)
    #         if random_number < deltaC * A[i, j]:
    #             sub_node_set.add(i)
    #             sub_node_set.add(j)
    #             A[i, j] = theta * frobenius_norm / math.sqrt(s)
    #         else:
    #             A[i, j] = 0

    for i in range(A.shape[0]):
        start_idx = A.indptr[i]
        end_idx = A.indptr[i + 1]

        # 遍历当前行的非零元素
        for j in range(start_idx, end_idx):
            col_idx = A.indices[j]
            value = A.data[j]

            # 在这里执行相应的操作
            if value > delta:
                # 根据条件进行修改
                sub_node_set.add(i)
                sub_node_set.add(col_idx)
                A[i, col_idx] = max(value, pow(frobenius_norm, 2) / (s * value))
            else:
                random_number = random.uniform(0, 1)
                if random_number < deltaC * value:
                    sub_node_set.add(i)
                    sub_node_set.add(col_idx)
                    A[i, col_idx] = theta * frobenius_norm / math.sqrt(s)
                else:
                    A[i, col_idx] = 0

    A.eliminate_zeros()

    print("the number of subgraph's edges: ", A.nnz / original_edges_nums)
    print("the number of subgraph's nodes: ", len(sub_node_set))
    print("start compute pagerank")
    sys.stdout.flush()

    # 初始向量
    R = np.repeat(1.0 / n, n)
    # 跳转概率分布
    P = np.repeat(1.0 / n, n)

    # PageRank迭代
    c = 0.85
    tol = 1 / n / 10
    for _ in range(sys.maxsize):
        R_last = R
        R = c * (A @ R) + (1 - c) * P
        # 计算误差
        err = np.absolute(R - R_last).sum()
        if err < n * tol:
            # 保存结果
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"execution_time：{execution_time:.2f} s")
            print("the number of iterations: ", _ + 1)
            with open(output_path, 'w+') as file:
                for i in range(n):
                    file.write(f"{node_dict_t[i]}\t{R[i]:.17f}\n")
                break


# 使用networkit，元素采样
def elementSamplingWithNetworkit(graph_path, output_path, alpha, theta):
    node_set = set()
    # original_edges = []
    node_degree = {}
    with open(graph_path, 'r') as file:
        for line in file:
            if line.startswith('%') or line.startswith('#'):
                continue
            data = line.strip().split()
            from_node = int(data[0])
            to_node = int(data[1])
            # original_edges.append((from_node, to_node))
            node_set.add(from_node)
            node_set.add(to_node)
            if from_node in node_degree.keys():
                node_degree[from_node] += 1
            else:
                node_degree[from_node] = 1
            if to_node in node_degree.keys():
                node_degree[to_node] += 1
            else:
                node_degree[to_node] = 1

    node_dict = {element: index for index, element in enumerate(node_set)}  # 节点id:原图index
    node_dict_t = {index: element for index, element in enumerate(node_set)}  # 原图index:节点id

    del node_set
    gc.collect()
    print("start construct graph")
    sys.stdout.flush()

    frobenius_norm = 0
    # 原始图
    G = nk.Graph(directed=True, weighted=True, n=len(node_dict))
    with open(graph_path, 'r') as file:
        for line in file:
            if line.startswith('%') or line.startswith('#'):
                continue
            data = line.strip().split()
            from_node = int(data[0])
            to_node = int(data[1])
            from_node_index = node_dict[from_node]
            to_node_index = node_dict[to_node]
            G.addEdge(from_node_index, to_node_index, 1.0 / node_degree[from_node])
            G.addEdge(to_node_index, from_node_index, 1.0 / node_degree[to_node])
            frobenius_norm += math.pow(1.0 / node_degree[from_node], 2)
            frobenius_norm += math.pow(1.0 / node_degree[to_node], 2)

    # for from_node, to_node in original_edges:
    #     from_node_index = node_dict[from_node]
    #     to_node_index = node_dict[to_node]
    #     G.addEdge(from_node_index, to_node_index, 1.0 / node_degree[from_node])
    #     G.addEdge(to_node_index, from_node_index, 1.0 / node_degree[to_node])
    #     frobenius_norm += math.pow(1.0 / node_degree[from_node], 2)
    #     frobenius_norm += math.pow(1.0 / node_degree[to_node], 2)

    # del original_edges
    del node_degree
    gc.collect()

    # 原图的边数和节点数
    original_edges_num = G.numberOfEdges()
    frobenius_norm = math.sqrt(frobenius_norm)
    s = original_edges_num / alpha
    delta = theta * frobenius_norm / math.sqrt(s)
    deltaC = 1 / delta
    print(f"f_norm: {frobenius_norm}")
    print(f"s: {s}")
    print(f"delta: {delta}")
    print(f"deltaC: {deltaC}")
    sys.stdout.flush()

    start_time = time.time()

    count_edges = 0
    # need_removed = []
    need_add = []
    for node in G.iterNodes():
        for (out_node, weight) in G.iterNeighborsWeights(node):
            if weight > delta:
                count_edges += 1
                new_weight = max(weight, pow(frobenius_norm, 2) / (s * weight))
                need_add.append((node, out_node, new_weight))
                # G.setWeight(node, out_node, new_weight)
            else:
                random_number = random.uniform(0, 1)
                if random_number < deltaC * weight:
                    count_edges += 1
                    new_weight = theta * frobenius_norm / math.sqrt(s)
                    need_add.append((node, out_node, new_weight))
                    # G.setWeight(node, out_node, new_weight)
                # else:
                #     need_removed.append((node, out_node))
                    # G.setWeight(node, out_node, 0)
    # for (from_node, to_node) in need_removed:
    #     G.removeEdge(from_node, to_node)

    # del need_removed
    # gc.collect()

    del G
    gc.collect()

    print("start construct new graph")
    sys.stdout.flush()

    G2 = nk.Graph(directed=True, weighted=True, n=len(node_dict))
    for from_node, to_node, weight in need_add:
        G2.addEdge(from_node, to_node, weight)

    del need_add
    gc.collect()

    print("the number of subgraph's edges: ", count_edges / original_edges_num)
    sub_nodes = set()
    for node in G2.iterNodes():
        if G2.degreeIn(node) != 0 or G2.degreeOut(node) != 0:
            sub_nodes.add(node)
    print("the number of subgraph's nodes: ", len(sub_nodes))
    print("start compute pagerank")
    sys.stdout.flush()

    # 计算PageRank
    pr = nk.centrality.PageRank(G2, tol=1e-6, damp=0.85, distributeSinks=True)
    pr.maxIterations = 100
    pr.run()
    pagerank_scores = pr.scores()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"execution_time：{execution_time:.2f} s")
    print("the number of iterations: ", pr.numberOfIterations())

    # data_to_write = [(node_dict_t[node], pagerank) for node, pagerank in enumerate(pagerank_scores) if node in sub_nodes]
    # with open(output_path, 'w+') as file, ThreadPoolExecutor() as executor:
    #     file.writelines(executor.map(write_to_file, data_to_write))

    with open(output_path, 'w+') as file:
        for node, pagerank in enumerate(pagerank_scores):
            if node in sub_nodes:
                file.write(f"{node_dict_t[node]}\t{pagerank:.17f}\n")

    # # 求矩阵的F范数
    # f_norm = 0.0
    # for edge in G3.iterEdgesWeights():
    #     from_node, to_node, weight = edge
    #     f_norm += math.pow(weight, 2)
    # s = (1 - sampling_ratio) * G3.numberOfEdges()
    # delta = theta * math.sqrt(f_norm) / math.sqrt(s)
    # deltaC = 1 / delta
    #
    # print(f"f_norm: {f_norm}")
    # print(f"s: {s}")
    # print(f"delta: {delta}")
    # print(f"deltaC: {deltaC}")
    #
    # # 元素抽样
    # G4 = nk.Graph(directed=True, weighted=True)
    # count = 0
    # for edge in G3.iterEdgesWeights():
    #     from_node, to_node, weight = edge
    #     if weight < delta:
    #         random_number = random.uniform(0, 1)
    #         if random_number < deltaC * weight:
    #             G4.addEdge(from_node, to_node, delta, addMissing=True)
    #         else:
    #             count += 1
    #     else:
    #         new_weight = max(weight, f_norm / (s * weight))
    #         G4.addEdge(from_node, to_node, new_weight, addMissing=True)
    # print(f"the number of the deleted edges: {count}")
    # print(f"the proportion of deleting edges : {count / original_edges_num}")
    # print(f"After Edge Sparsification, the number of nodes: {G4.numberOfNodes()}")
    #
    # # 计算PageRank
    # pr = nk.centrality.PageRank(G4, tol=1e-6, damp=0.85)
    # pr.run()
    # pagerank_scores = pr.scores()
    # print("the number of iterations: ", pr.numberOfIterations())
    #
    # with open(output_path, 'w+') as file:
    #     for node, pagerank in enumerate(pagerank_scores):
    #         file.write(f"{node + 1}\t{pagerank:.17f}\n")


# 使用networkit，向量采样
def vectorSampling(graph_path, output_path, sampling_ratio):
    import numpy as np

    node_set = set()
    # original_edges = []
    with open(graph_path, 'r') as file:
        for line in file:
            if line.startswith('%') or line.startswith('#'):
                continue
            data = line.strip().split()
            from_node = int(data[0])
            to_node = int(data[1])
            # original_edges.append((from_node, to_node))
            node_set.add(from_node)
            node_set.add(to_node)

    node_dict = {element: index for index, element in enumerate(node_set)}  # 节点id:原图index
    node_dict_t = {index: element for index, element in enumerate(node_set)}  # 原图index:节点id]

    del node_set
    gc.collect()

    print("start construct graph")
    sys.stdout.flush()

    # 原始图
    G = nk.Graph(directed=True, n=len(node_dict))
    with open(graph_path, 'r') as file:
        for line in file:
            if line.startswith('%') or line.startswith('#'):
                continue
            data = line.strip().split()
            from_node = int(data[0])
            to_node = int(data[1])
            from_node_index = node_dict[from_node]
            to_node_index = node_dict[to_node]
            G.addEdge(from_node_index, to_node_index)
            G.addEdge(to_node_index, from_node_index)

    # for from_node, to_node in original_edges:
    #     from_node_index = node_dict[from_node]
    #     to_node_index = node_dict[to_node]
    #     G.addEdge(from_node_index, to_node_index)
    #     G.addEdge(to_node_index, from_node_index)

    # 原图的边数
    original_edges_num = G.numberOfEdges()

    # del original_edges
    # gc.collect()

    print("start compute p distribution")
    sys.stdout.flush()

    # 计算p分布
    p = {}
    sum_p = 0
    for node in G.iterNodes():
        for in_node in G.iterInNeighbors(node):
            weight = math.pow(1 / G.degreeOut(in_node), 2)
            sum_p += weight
            if node in p.keys():
                p[node] += weight
            else:
                p[node] = weight
    # for key, value in p.items():
    #     p[key] = value / sum_p

    # 归一化概率
    probabilities = np.array(list(p.values())) / np.sum(list(p.values()))

    print("start sampling")
    sys.stdout.flush()

    start_time = time.time()

    # 抽列（按照p分布来抽）
    nodes = np.array(list(p.keys()))
    # probabilities = np.array(list(normalized_p.values()))
    sampled_node_index = np.random.choice(nodes, size=int(len(node_dict_t) * sampling_ratio), p=probabilities, replace=False)  # 按照分布进行采样

    # # 抽列（均匀抽）
    # sampled_node_index = np.random.choice(range(0, len(node_dict_t)), size=int(len(node_dict_t) * sampling_ratio), replace=False)

    # need_remove = []
    # for node in G.iterNodes():
    #     if node not in sampled_node_index:
    #         for in_node in G.iterInNeighbors(node):
    #             need_remove.append((in_node, node))
    # for (in_node, node) in need_remove:
    #     G.removeEdge(in_node, node)

    print("start construct graph")
    sys.stdout.flush()

    need_add = []
    sub_nodes = set()
    for node in G.iterNodes():
        if node in sampled_node_index:
            for in_node in G.iterInNeighbors(node):
                need_add.append((in_node, node))
                sub_nodes.add(in_node)
                sub_nodes.add(node)
    G2 = nk.Graph(directed=True, n=len(node_dict))
    for in_node, to_node in need_add:
        G2.addEdge(in_node, to_node)

    print(f"the proportion of subgraph edges : {len(need_add) / original_edges_num}")

    # # 构建子图与原图节点之间的对应关系
    # sub_node_set = set()
    # for node in G.iterNodes():
    #     if node in sampled_node_index:
    #         sub_node_set.add(node)
    #         for in_node in G.iterInNeighbors(node):
    #             sub_node_set.add(in_node)
    # sub_node_dict = {element: index for index, element in enumerate(sub_node_set)}  # 原图index:子图index
    # sub_node_dict_t = {index: element for index, element in enumerate(sub_node_set)}  # 子图index:原图index

    # subGraph = nk.Graph(n=len(sub_node_set), directed=True)
    # for node in G.iterNodes():
    #     if node in sampled_node_index:
    #         for in_node in G.iterInNeighbors(node):
    #             subGraph.addEdge(sub_node_dict[in_node], sub_node_dict[node])

    # print(f"the proportion of subgraph edges : {(original_edges_num - len(need_remove)) / original_edges_num}")

    # del need_remove
    # gc.collect()

    # sub_nodes = set()
    # for node in G2.iterNodes():
    #     if G2.degree(node) != 0:
    #         sub_nodes.add(node)
    print(f"After Edge Sparsification, the number of nodes: {len(sub_nodes)}")
    sys.stdout.flush()

    # 计算pagerank
    pr = nk.centrality.PageRank(G2, tol=1e-6, damp=0.85, distributeSinks=True)
    pr.maxIterations = 100
    pr.run()
    pagerank_scores = pr.scores()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"execution_time：{execution_time:.2f} s")
    print("the number of iterations: ", pr.numberOfIterations())
    sys.stdout.flush()

    # data_to_write = [(node_dict_t[node], pagerank) for node, pagerank in enumerate(pagerank_scores) if node in sub_nodes]
    # with open(output_path, 'w+') as file, ThreadPoolExecutor() as executor:
    #     file.writelines(executor.map(write_to_file, data_to_write))

    with open(output_path, 'w+') as file:
        for node, pagerank in enumerate(pagerank_scores):
            if node in sub_nodes:
                file.write(f"{node_dict_t[node]}\t{pagerank:.17f}\n")


# 使用networkit，向量抽样，抽行加抽列
def vectorSamplingBoth(graph_path, output_path, sampling_ratio):
    import numpy as np

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

    node_dict = {element: index for index, element in enumerate(node_set)}  # 节点id:原图index
    node_dict_t = {index: element for index, element in enumerate(node_set)}  # 原图index:节点id]

    del node_set
    gc.collect()

    print("start construct graph")
    sys.stdout.flush()

    # 原始图
    G = nk.Graph(directed=True, n=len(node_dict))
    with open(graph_path, 'r') as file:
        for line in file:
            if line.startswith('%') or line.startswith('#'):
                continue
            data = line.strip().split()
            from_node = int(data[0])
            to_node = int(data[1])
            from_node_index = node_dict[from_node]
            to_node_index = node_dict[to_node]
            G.addEdge(from_node_index, to_node_index)
            G.addEdge(to_node_index, from_node_index)


    # 原图的边数
    original_edges_num = G.numberOfEdges()

    print("start compute p distribution")
    sys.stdout.flush()

    # 计算p分布
    p_col = {}
    p_row = {}
    for node in G.iterNodes():
        for in_node in G.iterInNeighbors(node):
            weight = math.pow(1 / G.degreeOut(in_node), 2)
            if node in p_col.keys():
                p_col[node] += weight
            else:
                p_col[node] = weight
        for _ in G.iterNeighbors(node):
            weight = math.pow(1 / G.degreeOut(node), 2)
            if node in p_row.keys():
                p_row[node] += weight
            else:
                p_row[node] = weight

    p = {}
    for key, value in p_col.items():
        p[key] = p_col[key] * p_row[key]

    # 归一化概率
    probabilities = np.array(list(p.values())) / np.sum(list(p.values()))

    print("start sampling")
    sys.stdout.flush()

    start_time = time.time()

    # 抽列（按照p分布来抽）
    nodes = np.array(list(p.keys()))
    # probabilities = np.array(list(normalized_p.values()))
    c = int(len(node_dict_t) * sampling_ratio)
    sampled_node_index = np.random.choice(nodes, size=c, p=probabilities, replace=False)  # 按照分布进行采样

    print("start construct graph")
    sys.stdout.flush()

    need_add_edges = {}
    sub_nodes = set()
    for node in G.iterNodes():
        if node in sampled_node_index:
            sub_nodes.add(node)
            for in_node in G.iterInNeighbors(node):
                sub_nodes.add(in_node)
                if (in_node, node) not in need_add_edges.keys():
                    weight = 1 / G.degreeOut(in_node) / math.sqrt(c * p[node])
                    need_add_edges[(in_node, node)] = weight
    for node in G.iterNodes():
        if node in sampled_node_index:
            for to_node in G.iterNeighbors(node):
                sub_nodes.add(to_node)
                if (node, to_node) not in need_add_edges.keys():
                    weight = 1 / G.degreeOut(node) / math.sqrt(c * p[to_node])
                    need_add_edges[(node, to_node)] = weight

    G2 = nk.Graph(directed=True, weighted=True,n=len(node_dict))
    for key, value in need_add_edges.items():
        in_node = key[0]
        to_node = key[1]
        G2.addEdge(in_node, to_node, value)

    print(f"the proportion of subgraph edges : {len(need_add_edges) / original_edges_num}")

    print(f"After Edge Sparsification, the number of nodes: {len(sub_nodes)}")
    sys.stdout.flush()

    # 计算pagerank
    pr = nk.centrality.PageRank(G2, tol=1e-6, damp=0.85, distributeSinks=True)
    pr.maxIterations = 100
    pr.run()
    pagerank_scores = pr.scores()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"execution_time：{execution_time:.2f} s")
    print("the number of iterations: ", pr.numberOfIterations())
    sys.stdout.flush()

    with open(output_path, 'w+') as file:
        for node, pagerank in enumerate(pagerank_scores):
            if node in sub_nodes:
                file.write(f"{node_dict_t[node]}\t{pagerank:.17f}\n")


# 使用networkit，向量抽样，抽行加抽列
def vectorSamplingBothWithMatrix(graph_path, output_path, sampling_ratio):
    import numpy as np
    import scipy as sp
    import scipy.sparse

    # 因为存在一些数据集的id不是自增的，所以节点id的最大值可能不是节点总数
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

    # 计算p分布（使用行和列范数）
    # col_sums = A.power(2).sum(axis=0)
    # row_sums = A.power(2).sum(axis=1)
    # p = np.multiply(col_sums, row_sums)
    # probabilities = p / sum(p)

    # 列范数抽
    # total_sums = A.power(2).sum()
    # col_sums = A.power(2).sum(axis=0)
    # col_distribution = col_sums / total_sums

    # 行范数抽
    total_sums = A.power(2).sum()
    row_sums = A.power(2).sum(axis=1)
    row_distribution = row_sums / total_sums

    start_time = time.time()

    c = int(len(node_dict_t) * sampling_ratio)
    sampled_index = np.random.choice(A.shape[1], size=c, replace=False, p=row_distribution)
    sampled_index.sort()
    # 下面做法太占内存
    # A_col = A / probabilities
    # A_row = A / probabilities[:, np.newaxis]
    C = A[:, sampled_index]  # 列向量，n*c
    C = sp.sparse.csc_array(C)
    R = A[sampled_index, :]  # 行向量，c*n

    print("the number of C edges: ", C.nnz / original_edges_num)
    sys.stdout.flush()

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

    r = np.repeat(1.0 / n, n)
    P = np.repeat(1.0 / n, n)
    alpha = 0.85
    r = r @ C
    tol = 1 / n / 10
    for _ in range(sys.maxsize):
        r_last = r
        r = r @ R
        # r = r / np.linalg.norm(r, ord=1)
        r = alpha * r + (1 - alpha) * P
        r = r @ C
        # 计算误差
        err = np.absolute(r - r_last).sum()
        if err < c * tol:
            r = r @ R
            r = r / np.linalg.norm(r, ord=1)
            r = alpha * r + (1 - alpha) * P
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"execution_time：{execution_time:.2f} s")
            print("the number of iterations: ", _ + 1)
            # 保存结果
            with open(output_path, 'w+') as file:
                for i in range(len(node_dict_t)):
                    file.write(f"{node_dict_t[i]}\t{r[i]:.17f}\n")
                break


# 向量采样，使用SVD的方式替代原转移矩阵的迭代
def vectorSamplingBySVD(graph_path, output_path, node_num, sampling_ratio):
    import numpy as np
    import scipy as sp
    import scipy.sparse
    import scipy.sparse.linalg

    # 因为存在一些数据集的id不是自增的，所以节点id的最大值可能不是节点总数
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

    # 加载数据
    row, col = [], []
    with open(graph_path, 'r') as file:
        for line in file:
            if line.startswith('%') or line.startswith('#'): continue
            data = line.strip().split()
            from_node = int(data[0])
            to_node = int(data[1])
            row.append(node_dict[from_node])
            col.append(node_dict[to_node])
            row.append(node_dict[to_node])
            col.append(node_dict[from_node])
    data = np.ones(len(row))

    # 生成转移矩阵A
    n = node_num
    A = sp.sparse.coo_array((data, (row, col)), shape=(n, n), dtype=float)
    S = A.sum(axis=1)
    S[S != 0] = 1.0 / S[S != 0]
    Q = sp.sparse.csr_array(sp.sparse.spdiags(S.T, 0, *A.shape))
    A = Q @ A

    # 抽列（先按照均匀抽样）
    sub_graph_num = int(n * sampling_ratio)
    selected_node_indices = np.random.choice(np.arange(0, n), size=sub_graph_num, replace=False)
    selected_sub_matrix = A[:, selected_node_indices]
    print("the number of subgraph's edges: ", selected_sub_matrix.nnz / A.nnz)
    U, S, VT = sp.sparse.linalg.svds(selected_sub_matrix, k=round(math.sqrt(sub_graph_num)), which='LM')
    U = sp.sparse.csr_array(U)
    UT = U.transpose()
    print("UUT")
    sys.stdout.flush()
    temp_matrix = U @ UT
    print("U @ UT")
    sys.stdout.flush()
    final_matrix = temp_matrix @ A

    # 初始向量
    R = np.repeat(1.0 / n, n)

    # 跳转概率分布
    P = np.repeat(1.0 / n, n)

    # PageRank迭代
    alpha = 0.85
    tol = 1e-6
    for _ in range(sys.maxsize):
        R_last = R
        R = alpha * (R @ final_matrix) + (1 - alpha) * P
        # 计算误差
        err = np.absolute(R - R_last).sum()
        if err < n * tol:
            print("the number of iterations: ", _)
            # 归一化
            total_sum = np.sum(R)
            normalized_R = R / total_sum
            # 保存结果
            with open(output_path, 'w+') as file:
                for i in range(0, n):
                    file.write(f"{node_dict_t[i]}\t{normalized_R[i]:.17f}\n")
                break


# 向量采样，使用SVD的方式替代原转移矩阵的迭代，抽行加抽列
def vectorSamplingBothBySVD(graph_path, output_path, node_num, sampling_ratio, gamma):
    import numpy as np
    import scipy as sp
    import scipy.sparse
    import scipy.sparse.linalg

    # 因为存在一些数据集的id不是自增的，所以节点id的最大值可能不是节点总数
    node_set = set()
    with open(graph_path, 'r') as file:
        for line in file:
            if line.startswith('%') or line.startswith('#'): continue
            data = line.strip().split()
            from_node = int(data[0])
            to_node = int(data[1])
            node_set.add(from_node)
            node_set.add(to_node)
    node_dict = {element: index for index, element in enumerate(node_set)}
    node_dict_t = {index: element for index, element in enumerate(node_set)}

    # 加载数据
    row, col = [], []
    with open(graph_path, 'r') as file:
        for line in file:
            if line.startswith('%') or line.startswith('#'): continue
            data = line.strip().split()
            from_node = int(data[0])
            to_node = int(data[1])
            row.append(node_dict[from_node])
            col.append(node_dict[to_node])
            row.append(node_dict[to_node])
            col.append(node_dict[from_node])
    data = np.ones(len(row))

    # 生成转移矩阵A
    n = node_num
    A = sp.sparse.coo_array((data, (row, col)), shape=(n, n), dtype=float)
    S = A.sum(axis=1)
    S[S != 0] = 1.0 / S[S != 0]
    Q = sp.sparse.csr_array(sp.sparse.spdiags(S.T, 0, *A.shape))
    A = Q @ A

    # 抽行加抽列（先按照均匀抽样）
    sub_graph_num = int(n * sampling_ratio)
    selected_node_indices_col = np.random.choice(np.arange(0, n), size=sub_graph_num, replace=False)
    selected_node_indices_row = np.random.choice(np.arange(0, n), size=sub_graph_num, replace=False)
    C = A[:, selected_node_indices_col]
    selected_sub_matrix = A[selected_node_indices_row][:, selected_node_indices_col]
    print("the number of subgraph's edges: ", selected_sub_matrix.nnz / A.nnz)

    # 进行SVD分解
    tmp = sp.sparse.csr_matrix(selected_sub_matrix).todense()
    U, Sigma, V = np.linalg.svd(tmp, full_matrices=True)
    sp_V = sp.sparse.csr_array(V)
    sp_sigma_v = sp.sparse.csr_array(sp.sparse.spdiags(Sigma.T, 0, *sp_V.shape))
    H = C @ sp.sparse.linalg.inv(sp_sigma_v)

    # 低秩近似矩阵
    # eigenvalues, eigenvectors = sp.sparse.linalg.eigs(selected_sub_matrix, k=sub_graph_num-2)
    # fro_w = math.pow(sp.sparse.linalg.norm(selected_sub_matrix, 'fro'), 2)
    # t = np.sum(eigenvalues > gamma * fro_w)
    # l = min(round(math.sqrt(sub_graph_num)), t)
    l = round(math.sqrt(sub_graph_num))
    H_l = H[:, range(0, l)]
    print("H_l")
    sys.stdout.flush()

    # 计算近似转移矩阵
    temp_matrix = H_l @ H_l.transpose()
    print("H_l @ H_l_T")
    sys.stdout.flush()
    final_matrix = temp_matrix @ A

    # 初始向量
    R = np.repeat(1.0 / n, n)

    # 跳转概率分布
    P = np.repeat(1.0 / n, n)

    # PageRank迭代
    alpha = 0.85
    tol = 1e-6
    for _ in range(sys.maxsize):
        R_last = R
        R = alpha * (R @ final_matrix) + (1 - alpha) * P
        # 计算误差
        err = np.absolute(R - R_last).sum()
        if err < n * tol:
            print("the number of iterations: ", _)
            # 归一化
            total_sum = np.sum(R)
            normalized_R = R / total_sum
            # 保存结果
            with open(output_path, 'w+') as file:
                for i in range(0, n):
                    file.write(f"{node_dict_t[i]}\t{normalized_R[i]:.17f}\n")
                break


args = sys.argv
algorithm = args[1]
graph_path = args[2]
output_path = args[3]

if algorithm == "ESTheta":
    theta = float(args[4])
    # computeSubPageRankByTheta(graph_path, output_path, theta)
    computeSubPageRankByThetaWithNetworkit2(graph_path, output_path, theta)
elif algorithm == "ESTau":
    tau = float(args[4])
    # computeSubPageRankByTau(graph_path, output_path, tau)
    computeSubPageRankByTauWithNetworkit(graph_path, output_path, tau)
elif algorithm == "ApproxRank":
    sampling_ratio = float(args[4])
    node_num = int(args[5])
    approxRankWithNetworkit(graph_path, output_path, node_num, sampling_ratio)
elif algorithm == "ElementSampling":
    alpha = float(args[4])
    theta = float(args[5])
    # elementSampling(graph_path, output_path, alpha, theta)
    elementSamplingWithNetworkit(graph_path, output_path, alpha, theta)
elif algorithm == "VectorSampling":
    sampling_ratio = float(args[4])
    # vectorSampling(graph_path, output_path, sampling_ratio)
    # vectorSamplingBoth(graph_path, output_path, sampling_ratio)
    vectorSamplingBothWithMatrix(graph_path, output_path, sampling_ratio)
elif algorithm == "VectorSamplingSVD":
    sampling_ratio = float(args[4])
    node_num = int(args[5])
    vectorSamplingBySVD(graph_path, output_path, node_num, sampling_ratio)
elif algorithm == "VectorSamplingSVDBoth":
    sampling_ratio = float(args[4])
    node_num = int(args[5])
    epsilon = float(args[6])
    gamma = epsilon / 100
    vectorSamplingBothBySVD(graph_path, output_path, node_num, sampling_ratio, gamma)
elif algorithm == "GroundTruth":
    computeGlobalPageRankWithNetworkit(graph_path, output_path)

