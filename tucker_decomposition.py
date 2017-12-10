import numpy as np
import scipy.sparse.linalg as ln


# noinspection PyStringFormat,PyCompatibility,PyTypeChecker
def tucker_decomposition(matrix, target_dim):
    current_dim = matrix.shape
    dim_length = len(current_dim)
    temp_mat = matrix
    max_iterations = 50
    delta_threshold = 0.0001
    u = []

    for i in range(dim_length):
        if current_dim[i] < target_dim[i]:
            raise
        else:
            u.append(np.random.rand(current_dim[i], target_dim[i]))

    u_temp = 0
    norm_m = np.linalg.norm(temp_mat)
    prev_fit = 0
    core = 0
    for iteration in range(max_iterations):
        for idx in range(dim_length):
            u_temp = __comp_ubut_nth(temp_mat, u, idx)
            u[idx] = __svd_n(u_temp, idx, target_dim[idx])

        core = __tensor_multiplication(u_temp, u, dim_length - 1, True)
        norm_residual = np.sqrt(abs(pow(norm_m, 2) - pow(np.linalg.norm(core), 2)))
        fit = 1 - norm_residual / norm_m
        fit_delta = abs(fit - prev_fit)
        print(("Iteration %d : fit %e fit delta %e;" % (iteration, fit, fit_delta)))

        if iteration >= 1 and fit_delta < delta_threshold:
            break
        prev_fit = fit

    return core, u


def __tensor_multiplication(matrix, u, n, trans=False):
    current_dim = matrix.shape
    dim_length = len(current_dim)
    temp_mat = matrix
    rot_idx = np.r_[n:dim_length, 0:n]
    invrot_idx = np.r_[dim_length - n:dim_length, 0:dim_length - n]
    temp_mat = np.transpose(temp_mat, rot_idx)
    temp_shape = list(temp_mat.shape)
    temp_mat = temp_mat.reshape(temp_shape[0], int(temp_mat.size / temp_shape[0]))

    if trans:
        temp_mat = np.dot(u[n].transpose(), temp_mat)
        temp_shape[0] = u[n].shape[1]
    else:
        temp_mat = np.dot(u[n], temp_mat)
        temp_shape[0] = u[n].shape[0]
    temp_mat = temp_mat.reshape(temp_shape)
    temp_mat = np.transpose(temp_mat, invrot_idx)

    return temp_mat


def __comp_ubut_nth(matrix, u, n):
    idx = np.r_[0:n, n + 1:len(matrix.shape)]
    temp_mat = matrix

    for i in idx:
        temp_mat = __tensor_multiplication(temp_mat, u, i, True)

    return temp_mat


# noinspection PyTypeChecker
def __svd_n(matrix, n, k):
    curr_dim = matrix.shape
    dim_length = len(curr_dim)
    temp_mat = matrix
    rot_idx = np.r_[n, n + 1:dim_length, 0:n]
    temp_mat = np.transpose(temp_mat, rot_idx)
    rot_idx = temp_mat.shape
    temp_mat = np.reshape(temp_mat, [rot_idx[0], int(temp_mat.size / rot_idx[0])])

    if curr_dim[n] > k:
        u, s, v_t = ln.svds(np.dot(temp_mat, temp_mat.transpose()), k)
        return u
    if curr_dim[n] == k:
        return np.identity(k)
    raise


def recover(core, u):
    for i in range(len(core.shape)):
        core = __tensor_multiplication(core, u, i, False)

    return core
