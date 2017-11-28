import numpy as np


def unfold(tensor, n_mode):
    row_count, col_count = __get_unfolding_matrix_size(tensor, n_mode)
    result = np.zeros([row_count, col_count])

    mode_order = __get_unfolding_mode_order(tensor, n_mode)
    stride = __get_unfolding_stride(tensor, mode_order)

    for row in range(row_count):
        for col in range(col_count):
            i = __get_tensor_indices(row, col, tensor, n_mode, mode_order, stride)
            result[row, col] = tensor[tuple(map(int, i))]

    return result


def __get_unfolding_mode_order(tensor, n):
    return [i for i in range(n + 1, tensor.ndim)] + [i for i in range(n)]


def __get_unfolding_stride(tensor, mode_order):
    stride = [0 for x in range(tensor.ndim)]
    stride[mode_order[tensor.ndim - 2]] = 1

    for i in range(tensor.ndim - 3, -1, -1):
        stride[mode_order[i]] = (tensor.shape[mode_order[i + 1]] * stride[mode_order[i + 1]])

    return stride


def __get_tensor_indices(row, col: int, tensor, n, mode_order, stride):
    i = [0 for x in range(tensor.ndim)]
    i[n] = row
    i[mode_order[0]] = col / stride[mode_order[0]]

    for k in range(1, tensor.ndim - 1):
        i[mode_order[k]] = ((col % stride[mode_order[k - 1]]) / stride[mode_order[k]])

    return i


def __get_unfolding_matrix_size(tensor, n):
    row_count = tensor.shape[n]
    col_count = 1

    for i in range(tensor.ndim):
        if i != n:
            col_count *= tensor.shape[i]

    return row_count, col_count
