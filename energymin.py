import numpy as np
from numpy import linalg
from scipy.special import binom


def calc_n(row_index, col_index, num_points, degree):
    sum_ = 0
    for k in range(degree+1):
        n1 = num_points - degree
        k1 = row_index - k
        n2 = 2 * (num_points - degree)
        k2 = row_index + col_index - k
        if (n1 >= 0 and 0 <= k1 <= n1) and (n2 >= 0 and 0 <= k2 <= n2):
            sum_ += (-1) ** (degree - k) * binom(degree, k) * binom(n1, k1) / binom(n2, k2)
    return sum_


def sum_ns(i, n, m, js, coeffs):
    return sum(coeff*calc_n(i, j, n, m) for j, coeff in zip(js, coeffs))


def construct_e1_matrix(n):
    n_matrix = np.zeros((n-1, n-1))
    for i in range(1, n):
        for j in range(n-1):
            n_matrix[i-1][j] = sum_ns(i, n, 1, [j, j+1], [1, -1])
    return n_matrix


def construct_e2_matrix(n):
    n_matrix = np.zeros((n-1, n-1))
    for i in range(1, n):
        n_matrix[i-1][0] = calc_n(i, 1, n, 2) - 2*calc_n(i, 0, n, 2)
        for j in range(n-3):
            n_matrix[i-1][j+1] = calc_n(i, j, n, 2) - 2*calc_n(i, j+1, n, 2) + calc_n(i, j+2, n, 2)
        n_matrix[i-1][n-2] = calc_n(i, n-3, n, 2) - 2*calc_n(i, 0, n-2, 2)
    return n_matrix


def construct_n3_matrix(n):
    n_matrix = np.zeros((n-1, n-1))
    for i in range(1, n):
        n_matrix[i-1][0] = 3*calc_n(i, 0, n, 3) - calc_n(i, 1, n, 3)
        n_matrix[i-1][1] = -3*calc_n(i, 0, n, 3) + 3*calc_n(i, 1, n, 3) - calc_n(i, 2, n, 3)
        for j in range(n-5):
            n_matrix[i-1][j+2] = sum_ns(i, n, 3, range(j, j+5), [1, -3, 3, -1])
        n_matrix[i-1][n-3] = calc_n(i, n-5, n, 3) - 3*calc_n(i, n-4, n, 3) + 3*calc_n(i, n-3, n, 3)
        n_matrix[i-1][n-2] = calc_n(i, n-4, n, 3) - 3*calc_n(i, n-3, n, 3)
    return n_matrix


def construct_b_matrix(n, m, p0, pn):
    b = np.zeros((n-1, len(p0)))
    for i in range(1, n):
        b[i-1] = (-1)**(m+1)*calc_n(i, 0, n, 1)*p0 + (-1)**m*calc_n(i, n-m, n, 1)*pn
    return b


def add_known_point(a, b, p, p_index):
    a_new = np.delete(a, p_index-1, 0)
    b_new = np.delete(b, p_index-1, 0)
    b_new = b_new - a_new[:, [p_index-1]] * (p * np.ones((len(a_new), 1)))
    a_new = np.delete(a_new, p_index-1, 1)
    return a_new, b_new
