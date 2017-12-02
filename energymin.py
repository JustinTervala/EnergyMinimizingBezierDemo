import numpy as np
from numpy import linalg
from scipy.special import binom


def calc_n(row_index, col_index, num_points, degree):
    sum_ = 0
    for k in range(degree+1):
        if 0 <= row_index - k <= num_points-degree:
            sum_ += ((-1) ** (degree - k) *
                     binom(degree, k) *
                     binom(num_points - degree, row_index - k) /
                     binom(2 * (num_points - degree), row_index + col_index - k))
    return binom(num_points - degree, col_index) * sum_


def delta(k, m):
    return (-1)**(m-k)*binom(m, k)


def inner_sum(i, j, n, m):
    return sum(delta(abs(j-k), m)*calc_n(i, k, n, m)
               for k in range(max(0, j-m), min(j, n-m-1)+1))


def construct_energy_min_matrix(n, m):
    n_matrix = np.zeros((n - 1, n - 1))
    for i in range(1, n):
        for j in range(1, n):
            n_matrix[i-1][j-1] = inner_sum(i, j, n, m)
    return n_matrix


def construct_b_matrix(n, m, p0, pn):
    b = np.zeros((n-1, len(p0)))
    for i in range(1, n):
        b[i-1] = (-1)**m*calc_n(i, 0, n, 1)*p0 + calc_n(i, n-m, n, 1)*pn
    return b


def add_known_point(a, b, p, p_index):
    a_new = np.delete(a, p_index-1, 0)  # Delete the rows
    b_new = np.delete(b, p_index-1, 0)
    b_new = b_new - a_new[:, [p_index-1]] * (p * np.ones((len(a_new), 1)))
    a_new = np.delete(a_new, p_index-1, 1)
    return a_new, b_new

