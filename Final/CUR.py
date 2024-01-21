import math
import sys
import gc
import time

def SVD_Trans(graph_path, output_path, node_num, sampling_ratio):
    import numpy as np
    import scipy as sp
    import scipy.sparse
    import scipy.sparse.linalg

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

    print("collect node_dict")
    sys.stdout.flush()
    del node_dict
    gc.collect()

    # Generate the transfer matrix A
    n = node_num
    A = sp.sparse.coo_array((data, (row, col)), shape=(n, n), dtype=float)

    print("collect row col")
    sys.stdout.flush()
    del row
    del col
    gc.collect()

    S = A.sum(axis=1)
    S[S != 0] = 1.0 / S[S != 0]
    Q = sp.sparse.csr_array(sp.sparse.spdiags(S.T, 0, *A.shape))
    A = Q @ A


    print("collect S Q")
    sys.stdout.flush()
    del S
    del Q
    gc.collect()

    # Calculate the column distribution
    total_sums = A.power(2).sum()
    col_sums = A.power(2).sum(axis=0)
    col_distribution = col_sums / total_sums

    start_time = time.time()

    # sample col
    sub_graph_num = int(n * sampling_ratio)
    selected_node_indices = np.random.choice(A.shape[1], size=sub_graph_num, replace=False, p=col_distribution)
    selected_sub_matrix = A[:, selected_node_indices]

    print("the number of selected_sub_matrix edges: ", selected_sub_matrix.nnz / A.nnz)
    print("collect A")
    sys.stdout.flush()
    del A
    gc.collect()

    # SVD decomposition
    U, S, VT = sp.sparse.linalg.svds(selected_sub_matrix, k=round(math.sqrt(sub_graph_num)), which='LM')
    sp_U = sp.sparse.csr_array(U)
    sp_VT = sp.sparse.csr_array(VT)

    print("collect U VT")
    sys.stdout.flush()
    del U
    del VT
    gc.collect()

    # compute PageRank
    R = np.repeat(1.0 / n, n)
    P = np.repeat(1.0 / n, n)

    alpha = 0.85
    tol = 1 / node_num / 10
    for _ in range(sys.maxsize):
        R_last = R
        R = R @ sp_U * S @ sp_VT
        t = np.repeat(1.0 / n, n)
        t[selected_node_indices] = R
        R = alpha * t + (1 - alpha) * P
        # calculate error
        err = np.absolute(R - R_last).sum()
        if err < n * sampling_ratio * tol:
            R = abs(R) / np.linalg.norm(R, ord=1)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"execution_time：{execution_time:.2f} s")
            print("the number of iterations: ", _ + 1)
            # Save results
            R = R / np.linalg.norm(R, ord=1)
            with open(output_path, 'w+') as file:
                for i in selected_node_indices:
                    file.write(f"{node_dict_t[i]}\t{R[i]:.17f}\n")
                break


def CUR_Trans(graph_path, output_path, node_num, sampling_ratio, row_sampling_ration):
    import numpy as np
    import scipy as sp
    import scipy.sparse
    import scipy.sparse.linalg

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

    print("collect node_dict")
    sys.stdout.flush()
    del node_dict
    gc.collect()

    # Generate the transfer matrix A
    n = node_num
    A = sp.sparse.coo_array((data, (row, col)), shape=(n, n), dtype=float)

    print("collect row col")
    sys.stdout.flush()
    del row
    del col
    gc.collect()

    S = A.sum(axis=1)
    S[S != 0] = 1.0 / S[S != 0]
    Q = sp.sparse.csr_array(sp.sparse.spdiags(S.T, 0, *A.shape))
    A = Q @ A

    print("collect S Q")
    sys.stdout.flush()
    del S
    del Q
    gc.collect()

    # Calculating row and column distributions
    total_sums = A.power(2).sum()
    col_sums = A.power(2).sum(axis=0)
    row_sums = A.power(2).sum(axis=1)
    col_distribution = col_sums / total_sums
    row_distribution = row_sums / total_sums

    print("collect col_sums row_sums")
    sys.stdout.flush()
    del col_sums
    del row_sums
    gc.collect()

    start_time = time.time()

    # The rows and columns are extracted according to the row-column distribution to form the U and R matrices
    sampled_col_index = np.random.choice(A.shape[1], size=int(node_num * sampling_ratio), replace=False, p=col_distribution)
    sampled_row_index = np.random.choice(A.shape[0], size=int(node_num * row_sampling_ration), replace=False, p=row_distribution)
    sampled_col_index.sort()
    sampled_row_index.sort()
    C = A[:, sampled_col_index]
    R = A[sampled_row_index, :]

    # TODO If you need to do scaling, you need the following code
    # # Divide the C and R matrices by the specific gravity
    # # row traversal
    # for i in range(R.shape[0]):
    #     start_idx = R.indptr[i]
    #     end_idx = R.indptr[i + 1]
    #     # 遍历当前行的非零元素
    #     for j in range(start_idx, end_idx):
    #         col_idx = R.indices[j]
    #         value = R.data[j]
    #         R[i, col_idx] = value / row_distribution[sampled_row_index[i]]
    #
    # # col traversal
    # C = sp.sparse.csc_array(C)
    # for j in range(C.shape[1]):
    #     start_idx = C.indptr[j]
    #     end_idx = C.indptr[j + 1]
    #     for i in range(start_idx, end_idx):
    #         row_idx = C.indices[i]
    #         value = C.data[i]
    #         C[row_idx, j] = value / col_distribution[sampled_col_index[j]]

    W = A[sampled_row_index][:, sampled_col_index]

    # TODO Compute the rank of W
    # rank_start_time = time.time()
    # _, s, _ = sp.sparse.linalg.svds(W, k=min(W.shape[0], W.shape[1]) - 1)
    # rank_sparse = np.sum(s > 1e-10)
    # print("rank:", rank_sparse)
    # rank_end_time = time.time()

    A_nnz = A.nnz

    print("collect A sampled_col_index sampled_row_index")
    del A
    del sampled_row_index
    del sampled_col_index
    gc.collect()

    print("the number of C edges: ", C.nnz / A_nnz)
    print("the number of R edges: ", R.nnz / A_nnz)
    print("the number of W edges: ", W.nnz / A_nnz)
    sys.stdout.flush()

    # SVD on W and then obtain U
    X, Z, YT = sp.sparse.linalg.svds(W, k=round(math.sqrt(min(W.shape[0], W.shape[1]))), which='LM')
    # rank_ratio = np.sum(Z) / np.sum(s)
    # print("k:", round(math.sqrt(min(W.shape[0], W.shape[1]))))
    # print("rank power ratio:", rank_ratio)
    Z = (1 / Z) ** 2
    U = YT.transpose() @ np.diag(Z) @ X.transpose()
    print("collect X Z YT")
    del X
    del Z
    del YT
    sys.stdout.flush()

    # compute PageRank
    sp_C = C
    sp_R = R
    R = np.repeat(1.0 / n, n)
    R = U @ sp_R @ R
    P = np.repeat(1.0 / n, n)
    alpha = 0.85
    tol = 1 / node_num / 10
    for _ in range(sys.maxsize):
        R_last = R
        R = R @ sp_C @ U @ sp_R
        R = alpha * R + (1 - alpha) * P
        # calculate error
        err = np.absolute(R - R_last).sum()
        if err < n * tol:
            R = abs(R) / np.linalg.norm(R, ord=1)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"execution_time：{execution_time:.2f} s")
            print("the number of iterations: ", _ + 1)
            # Save results
            R = R / np.linalg.norm(R, ord=1)
            with open(output_path, 'w+') as file:
                for i in range(len(node_dict_t)):
                    file.write(f"{node_dict_t[i]}\t{R[i]:.17f}\n")
                break


args = sys.argv
algorithm = args[1]
graph_path = args[2]
output_path = args[3]
sampling_ratio = float(args[4])
node_num = int(args[5])

if algorithm == "SVD_Trans":
    SVD_Trans(graph_path, output_path, node_num, sampling_ratio)
elif algorithm == "CUR_Trans":
    row_sampling_ration = float(args[6])
    CUR_Trans(graph_path, output_path, node_num, sampling_ratio, row_sampling_ration)