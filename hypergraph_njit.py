# Partial source: https://github.com/privet-kitty/hypergraph-discrepancy
from itertools import count
import numpy as np
from numba import njit

@njit()
def dot(A,b):
    """
    Simple matrix-vector dot product function.

    :param A: input matrix
    :type A: numpy.ndarray
    :param b: input vector
    :type b: numpy.ndarray
    :return: vector obtained from dot product
    """
    n, m = A.shape
    c = np.zeros(n) #np.float64
    for i in range(n):
        for j in range(m):
            c[i] += A[i][j] * b[j]
    return c

# 
@njit()
def calc_disc(incidence, coloring):
    """
    Given an incidence matrix and specific coloring, compute the discrepancy.

    :param incidence: incidence matrix (only 0's and 1's)
    :type incidence: numpy.ndarray
    :param coloring: coloring (only 1's and -1's)
    :type coloring: numpy.ndarray
    :return: discrepancy (int) of the incidence matrix, for this specific coloring
    """
    return np.max(np.absolute(dot(incidence,coloring)))

@njit(cache=True)
def cartesian_jit(N):
    """
    Function to compute all colorings of size N.

    :param N: length of colorings vectors
    :type N: int
    :return: matrix, with each row being a possible coloring (of size N)
    """
    arrays = [np.asarray([-1.0,1.0]) for i in range(N)]
    n = 1
    for x in arrays:
        n *= x.size
    out = np.zeros((n, len(arrays)))

    for i in range(len(arrays)):
        m = int(n / arrays[i].size)
        out[:n, i] = np.repeat(arrays[i], m)
        n //= arrays[i].size

    n = arrays[-1].size
    for k in range(len(arrays)-2, -1, -1):
        n *= arrays[k].size
        m = int(n / arrays[k].size)
        for j in range(1, arrays[k].size):
            out[j*m:(j+1)*m,k+1:] = out[0:m,k+1:]
    return out

@njit()
def find_opt_coloring(incidence):
    """
    Function to find the optimal coloring and associated (minimal) discrepancy
    for a particular incidence matrix.

    :param incidence: incidence matrix (only 0's and 1's)
    :type incidence: np.ndarray
    :return: optimal coloring (np.ndarray), associated discrepancy (int)
    and amount of colorings that lead to the associated (minimal) discrepancy
    """
    n = incidence.shape[1]
    min_disc = n
    opt_coloring = np.ones(n) #FIXME: dtype=int
    count = 0
    for c in cartesian_jit(n):
        if c[0] == 1: # fix first element of coloring
            disc = calc_disc(incidence, c)
            if disc < min_disc:
                min_disc = disc
                opt_coloring = c
                count = 0
            elif disc == min_disc:
                count +=1
    return np.asarray(opt_coloring), min_disc, count

@njit()
def calc_prefix_disc_simple(incidence):
    """
    Function to compute the (minimal) prefix discrepancy
    for a particular incidence matrix.

    :param incidence: incidence matrix (only 0's and 1's)
    :type incidence: np.ndarray
    :return: (minimal) prefix discrepancy
    """
    n = incidence.shape[1]
    min_disc = n
    opt_coloring = np.ones(n) #FIXME: dtype=int
    count = 0
    for c in cartesian_jit(n):
        if c[0] == 1: # fix first element of coloring
            prefix_UB = 0
            for i in range(n):
                disc = calc_disc(incidence[:,0:i+1], c)
                if disc > prefix_UB:
                    prefix_UB = disc
            if prefix_UB < min_disc:
                min_disc = prefix_UB
                opt_coloring = c
                count = 0
            elif prefix_UB == min_disc:
                count += 1
    return min_disc, count #, np.asarray(opt_coloring)

@njit()
def symmetry(incidence):
    """
    Function that computes the amount of "symmetry" of given
    incidence matrix.

    :param incidence: incidence matrix (only 0's and 1's)
    :type incidence: np.ndarray
    :return: "symmetry" (int) = var of row_count + var of col_count
    """
    m, n = incidence.shape
    row_count = np.zeros(m)
    col_count = np.zeros(n)
    for i in range(m):
        for j in range(n):
            row_count[i] += incidence[i][j]
    for j in range(n):
        for i in range(m):
            col_count[j] += incidence[j][i]
    return np.var(row_count) + np.var(col_count)