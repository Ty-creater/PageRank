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

    # load data
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

    # Generate the transfer matrix A
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

    # Number of edges of the original graph
    original_edges_num = A.nnz

    # TODO Use different distributions to pump nodes
    # Calculate p-distribution (using row and column f paradigms)
    col_sums = A.power(2).sum(axis=0)
    row_sums = A.power(2).sum(axis=1)
    p = np.multiply(col_sums, row_sums)
    probabilities = p / sum(p)

    # column f paradigms
    # total_sums = A.power(2).sum()
    # col_sums = A.power(2).sum(axis=0)
    # col_distribution = col_sums / total_sums

    # row f paradigms
    # total_sums = A.power(2).sum()
    # row_sums = A.power(2).sum(axis=1)
    # row_distribution = row_sums / total_sums

    start_time = time.time()

    c = int(len(node_dict_t) * sampling_ratio)
    # using row and column f paradigms
    sampled_index = np.random.choice(A.shape[1], size=c, replace=False, p=probabilities)
    # # column f paradigms
    # sampled_index = np.random.choice(A.shape[1], size=c, replace=False, p=col_distribution)
    # # row f paradigms
    # sampled_index = np.random.choice(A.shape[1], size=c, replace=False, p=row_distribution)
    # # uniform distribution
    # sampled_index = np.random.choice(A.shape[1], size=c, replace=False, p=None)
    sampled_index.sort()
    C = A[:, sampled_index]  # column vector，n*c
    C = sp.sparse.csc_array(C)
    R = A[sampled_index, :]  # row vector，c*n

    print("the number of C edges: ", C.nnz / original_edges_num)
    sys.stdout.flush()

    ## TODO If you want to do Scaling, you need the following code
    # probabilities = np.sqrt(probabilities * c)
    # # row traversal
    # for i in range(R.shape[0]):
    #     start_idx = R.indptr[i]
    #     end_idx = R.indptr[i + 1]
    #     # 遍历当前行的非零元素
    #     for j in range(start_idx, end_idx):
    #         col_idx = R.indices[j]
    #         value = R.data[j]
    #         R[i, col_idx] = value / probabilities[sampled_index[i]]
    #
    # # col traversal
    # for j in range(C.shape[1]):
    #     start_idx = C.indptr[j]
    #     end_idx = C.indptr[j + 1]
    #     for i in range(start_idx, end_idx):
    #         row_idx = C.indices[i]
    #         value = C.data[i]
    #         C[row_idx, j] = value / probabilities[sampled_index[j]]

    print("start compute pagerank")
    sys.stdout.flush()

    # Start iterative PageRank calculation
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
        # calculate error
        err = np.absolute(r - r_last).sum()
        if err < c * tol:
            r = C @ r
            r = (alpha / (1 + alpha)) * r + (1 - alpha) * P
            r = r / np.linalg.norm(r, ord=1)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"execution_time：{execution_time:.2f} s")
            print("the number of iterations: ", _ + 1)
            # Save results
            with open(output_path, 'w+') as file:
                for i in range(len(node_dict_t)):
                    file.write(f"{node_dict_t[i]}\t{r[i]:.17f}\n")
                break


args = sys.argv
graph_path = args[1]
output_path = args[2]
sampling_ratio = float(args[3])
T2(graph_path, output_path, sampling_ratio)

