import math
import numpy as np
import scipy as sp
import scipy.sparse
import networkit as nk
import random
import sys
import time
from networkit.centrality import PageRank
import gc


# 使用Networkit来计算原图的PageRank，用于实验的 ground truth
def computeGlobalPageRankWithNetworkit(graph_path, output_path):
    # TODO 因为不是所有的数据集的节点id都是按顺递增的，而Networkit构建的图需要严格递增，因此做一层节点映射（下同）。
    node_set = set()  # 节点集合（去重）
    original_edges = []  # 将边集加载到内存中，后续方法（如果出现是为了让方法之间的时间具有可比性，因为读磁盘与读内存速度差很多），这里该步骤可省去。
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
    node_dict = {element: index for index, element in enumerate(node_set)}  # 节点id:原图index
    node_dict_t = {index: element for index, element in enumerate(node_set)}  # 原图index:节点id

    # 构建图
    G = nk.Graph(directed=True, n=len(node_dict))
    for from_node, to_node in original_edges:
        from_node_index = node_dict[from_node]
        to_node_index = node_dict[to_node]
        G.addEdge(from_node_index, to_node_index)
        G.addEdge(to_node_index, from_node_index)

    # 初始化PageRank对象
    pr = nk.centrality.PageRank(G, tol=1 / len(node_dict) / 100, damp=0.85)
    # 开始计算PageRank
    pr.run()
    # 获取PageRank结果
    pagerank_scores = pr.scores()

    # 持久化
    with open(output_path, 'w+') as file:
        for node, pagerank in enumerate(pagerank_scores):
            file.write(f"{node_dict_t[node]}\t{pagerank:.17f}\n")


def PER_PR(graph_path, output_path, theta):
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

    start_time = time.time()  # 开始计时

    # 构图
    for from_node, to_node in original_edges:
        random_number = random.uniform(0, 1)
        if random_number < theta:  # 按比例
            count += 1
            G.addEdge(from_node - 1, to_node - 1, addMissing=True)
            G.addEdge(to_node - 1, from_node - 1)
    print("the number of selected_sub_graph edges: ", count / len(original_edges))
    del original_edges
    gc.collect()

    # 统计非孤立节点，并最后只持久化非孤立节点的PageRank
    sub_nodes = set()
    for node in G.iterNodes():
        if G.degreeIn(node) != 0 or G.degreeOut(node) != 0:
            sub_nodes.add(node)
    print(f"After Edge Sparsification, the number of nodes: {len(sub_nodes)}")
    sys.stdout.flush()

    # 计算PageRank
    pr = nk.centrality.PageRank(G, tol=1/G.numberOfNodes()/10, damp=0.85)
    pr.run()
    pagerank_scores = pr.scores()

    # 结束计时
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"execution_time：{execution_time:.2f} s")
    print("the number of iterations: ", pr.numberOfIterations())

    # 持久化
    with open(output_path, 'w+') as file:
        for node, pagerank in enumerate(pagerank_scores):
            if node in sub_nodes:
                file.write(f"{node+1}\t{pagerank:.17f}\n")


def DSPI(graph_path, output_path, alpha, theta):
    node_set = set()
    node_degree = {}  # 统计每个点的度
    with open(graph_path, 'r') as file:
        for line in file:
            if line.startswith('%') or line.startswith('#'):
                continue
            data = line.strip().split()
            from_node = int(data[0])
            to_node = int(data[1])
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
    # 构建原始图，并计算F范数所需的数据
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
    del node_degree
    gc.collect()

    # 原图的边数和节点数
    original_edges_num = G.numberOfEdges()
    # F范数
    frobenius_norm = math.sqrt(frobenius_norm)
    s = original_edges_num / alpha
    delta = theta * frobenius_norm / math.sqrt(s)
    deltaC = 1 / delta
    print(f"f_norm: {frobenius_norm}")
    print(f"s: {s}")
    print(f"delta: {delta}")
    print(f"deltaC: {deltaC}")
    sys.stdout.flush()

    # 开始计时
    start_time = time.time()

    count_edges = 0  # 采样的边数
    need_add = []  # 需要添加的边，因为networkit不能边遍历边对边进行修改，所以需要重新构图
    for node in G.iterNodes():
        for (out_node, weight) in G.iterNeighborsWeights(node):
            if weight > delta:
                count_edges += 1
                new_weight = max(weight, pow(frobenius_norm, 2) / (s * weight))
                need_add.append((node, out_node, new_weight))
            else:
                random_number = random.uniform(0, 1)
                if random_number < deltaC * weight:
                    count_edges += 1
                    new_weight = theta * frobenius_norm / math.sqrt(s)
                    need_add.append((node, out_node, new_weight))
    del G
    gc.collect()

    # 开始构建所需的子图
    print("start construct new graph")
    sys.stdout.flush()
    G2 = nk.Graph(directed=True, weighted=True, n=len(node_dict))
    for from_node, to_node, weight in need_add:
        G2.addEdge(from_node, to_node, weight)

    del need_add
    gc.collect()

    print("the number of subgraph's edges: ", count_edges / original_edges_num)
    sub_nodes = set()  # 同上
    for node in G2.iterNodes():
        if G2.degreeIn(node) != 0 or G2.degreeOut(node) != 0:
            sub_nodes.add(node)
    print("the number of subgraph's nodes: ", len(sub_nodes))
    print("start compute pagerank")
    sys.stdout.flush()

    # 计算PageRank
    pr = nk.centrality.PageRank(G2, tol=1e-6, damp=0.85, distributeSinks=True)
    # pr.maxIterations = 100
    pr.run()
    pagerank_scores = pr.scores()
    end_time = time.time()  # 结束计时
    execution_time = end_time - start_time
    print(f"execution_time：{execution_time:.2f} s")
    print("the number of iterations: ", pr.numberOfIterations())

    # 持久化
    with open(output_path, 'w+') as file:
        for node, pagerank in enumerate(pagerank_scores):
            if node in sub_nodes:
                file.write(f"{node_dict_t[node]}\t{pagerank:.17f}\n")


def ApproxRank(graph_path, output_path, node_num, sampling_ratio):
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

    start_time = time.time()  # 开始计时

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


def LocalPR(graph_path, output_path, sampling_num, edges_ration):
    node_set = set()
    original_edges_nums = 0
    with open(graph_path, 'r') as file:
        for line in file:
            if line.startswith('%') or line.startswith('#'):
                continue
            data = line.strip().split()
            from_node = int(data[0])
            to_node = int(data[1])
            node_set.add(from_node)
            node_set.add(to_node)
            original_edges_nums += 2
    node_dict = {element: index for index, element in enumerate(node_set)}
    node_dict_t = {index: element for index, element in enumerate(node_set)}

    print("start construct graph")
    sys.stdout.flush()
    del node_set
    gc.collect()

    # 原始图
    G = nk.Graph(directed=True, n=len(node_dict))
    with open(graph_path, 'r') as file:
        for line in file:
            if line.startswith('%') or line.startswith('#'):
                continue
            data = line.strip().split()
            from_node = int(data[0])  # 节点id
            to_node = int(data[1])
            from_node_index = node_dict[from_node]  # 原图index
            to_node_index = node_dict[to_node]
            G.addEdge(from_node_index, to_node_index)
            G.addEdge(to_node_index, from_node_index)

    print("start sampling")
    sys.stdout.flush()

    start_time = time.time()

    # 目标节点集（随机抽）
    target_nodes = np.random.choice(range(0, len(node_dict_t)), sampling_num, replace=False)

    result = {}  # 保存PageRank结果
    for u in target_nodes:
        if len(result) % 100 == 0:
            print("the number of completed nodes: ", len(result))
            sys.stdout.flush()
        pr_u = (1 - 0.85) / len(node_dict)
        layer = {u: 1}
        # 统计计算这个节点会访问多少条不同的边
        visited_edges = set()
        # 开始BFS扩展
        for _ in range(10):
            if len(result) == 0:
                print("the first node is computing, and layer is ", _)
                sys.stdout.flush()
            out_flag = False
            layer_next = {}
            for node, inf in layer.items():
                flag = False
                for in_node in G.iterInNeighbors(node):
                    visited_edges.add((in_node, node))
                    if in_node in layer_next.keys():
                        layer_next[in_node] += inf / G.degreeOut(in_node)
                    else:
                        layer_next[in_node] = inf / G.degreeOut(in_node)
                    if len(visited_edges) / original_edges_nums > edges_ration:  # 当前计算的节点，BFS扩展的边数达到要求之后，不在扩展
                        flag = True
                        break
                if flag:
                    out_flag = True
                    break
            pr_u += (1 - 0.85) / len(node_dict) * pow(0.85, _ + 1) * sum(layer_next.values())
            if out_flag:
                break
            layer = layer_next
        result[node_dict_t[u]] = pr_u

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"execution_time：{execution_time:.2f} s")

    with open(output_path, 'w+') as file:
        for node, pagerank in result.items():
            file.write(f"{node}\t{pagerank:.17f}\n")


def LPRAP(graph_path, output_path, sampling_num, edges_ration, T):
    node_set = set()
    original_edges_nums = 0
    with open(graph_path, 'r') as file:
        for line in file:
            if line.startswith('%') or line.startswith('#'):
                continue
            data = line.strip().split()
            from_node = int(data[0])
            to_node = int(data[1])
            node_set.add(from_node)
            node_set.add(to_node)
            original_edges_nums += 2
    node_dict = {element: index for index, element in enumerate(node_set)}
    node_dict_t = {index: element for index, element in enumerate(node_set)}

    print("start construct graph")
    sys.stdout.flush()
    del node_set
    gc.collect()

    # 原始图
    G = nk.Graph(directed=True)
    with open(graph_path, 'r') as file:
        for line in file:
            if line.startswith('%') or line.startswith('#'):
                continue
            data = line.strip().split()
            from_node = int(data[0])  # 节点id
            to_node = int(data[1])
            from_node_index = node_dict[from_node]  # 原图index
            to_node_index = node_dict[to_node]
            G.addEdge(from_node_index, to_node_index, addMissing=True)
            G.addEdge(to_node_index, from_node_index)

    print("start sampling")
    sys.stdout.flush()

    start_time = time.time()

    # 目标节点集
    target_nodes = np.random.choice(range(0, len(node_dict_t)), sampling_num, replace=False)

    result = {}
    visited_edges_nums = 0
    for u in target_nodes:
        if len(result) % 100 == 0:
            print("the number of completed nodes: ", len(result))
            sys.stdout.flush()
        pr_u = (1 - 0.85) / len(node_dict)
        layer = {u: 1}
        visited_edges = set()
        for _ in range(10):
            if len(result) == 0:
                print("the first node is computing, and layer is ", _)
                sys.stdout.flush()
            out_flag = False
            layer_next = {}
            for node, inf in layer.items():
                flag = False
                if inf * pow(0.85, _) < T:  # 如果小于给定阈值，也不在扩展该边
                    continue
                for in_node in G.iterInNeighbors(node):
                    visited_edges.add((in_node, node))
                    if in_node in layer_next.keys():
                        layer_next[in_node] += inf / G.degreeOut(in_node)
                    else:
                        layer_next[in_node] = inf / G.degreeOut(in_node)
                    if len(visited_edges) / original_edges_nums > edges_ration:
                        flag = True
                        break
                if flag:
                    out_flag = True
                    break
            pr_u += (1 - 0.85) / len(node_dict) * pow(0.85, _ + 1) * sum(layer_next.values())
            if out_flag:
                break
            layer = layer_next
        result[node_dict_t[u]] = pr_u
        visited_edges_nums += len(visited_edges)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"execution_time：{execution_time:.2f} s")

    with open(output_path, 'w+') as file:
        for node, pagerank in result.items():
            file.write(f"{node}\t{pagerank:.17f}\n")


args = sys.argv
algorithm = args[1]
graph_path = args[2]
output_path = args[3]

if algorithm == "GroundTruth":
    computeGlobalPageRankWithNetworkit(graph_path, output_path)
elif algorithm == "PER_PR":
    theta = float(args[4])
    PER_PR(graph_path, output_path, theta)
elif algorithm == "DSPI":
    alpha = float(args[4])
    theta = float(args[5])
    DSPI(graph_path, output_path, alpha, theta)
elif algorithm == "ApproxRank":
    sampling_ratio = float(args[4])
    node_num = int(args[5])
    ApproxRank(graph_path, output_path, node_num, sampling_ratio)
elif algorithm == "LocalPR":
    sampling_num = int(args[4])
    edges_ration = float(args[5])
    LocalPR(graph_path, output_path, sampling_num, edges_ration)
elif algorithm == "LPRAP":
    sampling_num = int(args[4])
    edges_ration = float(args[5])
    T = float(args[6])
    LPRAP(graph_path, output_path, sampling_num, edges_ration, T)
