# Partial source: https://github.com/privet-kitty/hypergraph-discrepancy
import numpy as np
from numba import njit

@njit
def dot(A,b):
    n, m = A.shape
    c = np.zeros(n) #np.float64
    for i in range(n):
        for j in range(m):
            c[i] += A[i][j] * b[j]
    return c

@njit()
def calc_discrepancy(incidence, coloring):
    return np.max(np.absolute(dot(incidence,coloring)))

@njit(cache=True)
def cartesian_jit(N):

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
    """Brute force search"""
    n = incidence.shape[1]
    min_disc = n
    opt_coloring = np.ones(n) #FIXME: dtype=int
    count = 0
    for c in cartesian_jit(n):
        if c[0] == 1: # fix first element of coloring
            disc = calc_discrepancy(incidence, c)
            if disc < min_disc:
                min_disc = disc
                opt_coloring = c
                count = 0
            elif disc == min_disc:
                count +=1
    return np.asarray(opt_coloring), min_disc, count

@njit
def calc_min_prefix_discrepancy(incidence):
    n = incidence.shape[1]
    max_disc = 0
    for i in range(n):
        opt_coloring, disc, count = find_opt_coloring(incidence[:,0:i+1])
        if disc > max_disc:
            max_disc = disc
    return max_disc

@njit
def calc_score_function(incidence):
    n = incidence.shape[1]
    max_disc = 0
    count_final = 0
    for i in range(n):
        opt_coloring, disc, count = find_opt_coloring(incidence[:,0:i+1])
        if disc > max_disc:
            max_disc = disc
            count_final = count
    return max_disc - 0.00001*count_final

@njit
def symmetry(incidence):
    max = 0
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