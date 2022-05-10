# Partial source: email from Lars
import numpy as np
from numba import njit

@njit(cache=True)
def cartesian_jit(M, d):

    numbers = np.array(list(range(-d,d+1)))
    arrays = [numbers for i in range(M)]
    
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

@njit(cache=True)
def bounded(v, d):
    return np.all((v >= -d) & (v <= d) == True)

@njit(cache=True)
def find_row_index(A, b):
    index = -1
    row_count = A.shape[0]
    for i in range(row_count):
        if (np.array_equal(A[i],b)):
            index = i
            break
    return index

@njit(cache=True)
def dynamic_table(A, d):
    
    m, n = A.shape
    D = np.empty((n+1,((2*d)+1)**m))
    # D.fill(-3)
    permutations = cartesian_jit(m, d)

    # Build table D[][] in bottom up manner
    for i in range(n + 1):
        # print(i)
        for v in permutations:
            v_i = find_row_index(permutations, v)
            if (i==0):
                if (np.count_nonzero(v)==0):
                    D[i][v_i] = True
                else:
                    D[i][v_i] = False
            elif (bounded(v - A[:,i-1], d) == True) and (D[i-1][find_row_index(permutations, v - A[:,i-1])] == True):
                    D[i][v_i] = True
            elif (bounded(v + A[:,i-1], d) == True) and (D[i-1][find_row_index(permutations, v + A[:,i-1])] == True):
                    D[i][v_i] = True
            else:
                D[i][v_i] = False
        
    boolean = False
    for v in permutations:
        v_i = find_row_index(permutations, v)
        if D[n][v_i] == True:
            boolean = True
            break
    
    return boolean

@njit(cache=True)
def calc_prefix_disc(incidence):
    prefix_disc = -1
    m, n = incidence.shape
    for d in range(2*m):
        if(dynamic_table(incidence, d) == True):
            prefix_disc = d
            break
    return prefix_disc