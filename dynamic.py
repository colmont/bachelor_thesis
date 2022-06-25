import numpy as np
from numba import njit

@njit(cache=True)
def cartesian_jit(M, d):
    """
    Function to compute all vectors v for dynamic table.

    :param M: number of sets in set system
    :type M: int
    :param d: constant d used in dynamic program
    :type d: int
    :return: matrix, with each row being a possible vector v (of size 2M+1)
    """
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
    """
    Function to determine if vector v is bounded by constant d.

    :param d: constant d used in dynamic program
    :type d: int
    :param v: vector v from dynamic program
    :type v: numpy.ndarray
    :return: boolean, true if vector v is bounded by constant d
    """
    return np.all((v >= -d) & (v <= d) == True)

@njit(cache=True)
def v_to_ind(d, v):
    """
    Function that turns vector v into an index using bit numbering.

    :param d: constant d used in dynamic program
    :type d: int
    :param v: vector v from dynamic program
    :type v: numpy.ndarray
    :return: index (int), so that vector v can be found quickly in dynamic table A
    """
    v = v + d
    base = (2*d)+1
    sum = 0
    N = len(v)
    for i in range(N):
        sum += v[i]*(base**(N-1-i))
    return int(sum)

@njit(cache=True)
def dynamic_table(A, d):
    """
    Using a dynamic table (see thesis), this function returns whether prefix-disc is smaller than d.

    :param A: incidence matrix
    :type A: numpy.ndarray
    :param d: constant d used in dynamic program
    :type d: int
    :return: boolean (true if prefix-disc(A) <= d)
    """
    m, n = A.shape
    D_1 = np.empty(((2*d)+1)**m)
    D_2 = np.empty(((2*d)+1)**m)
    permutations = cartesian_jit(m, d)

    # Build table D[][] in bottom up manner.
    for i in range(n + 1):
        for v in permutations:
            v_i = v_to_ind(d, v)
            if (i==0):
                if (np.count_nonzero(v)==0):
                    D_2[v_i] = True
                else:
                    D_2[v_i] = False
            elif (bounded(v - A[:,i-1], d) == True) and (D_1[v_to_ind(d, v - A[:,i-1])] == True):
                    D_2[v_i] = True
            elif (bounded(v + A[:,i-1], d) == True) and (D_1[v_to_ind(d, v + A[:,i-1])] == True):
                    D_2[v_i] = True
            else:
                D_2[v_i] = False
        D_1 = D_2
        D_2 = np.empty(((2*d)+1)**m)
        
    # Check whether there exists a v for which D[n,v] is true.
    boolean = False
    for v in permutations:
        v_i = v_to_ind(d, v)
        if D_1[v_i] == True:
            boolean = True
            break
    
    return boolean

@njit(cache=True)
def dynamic_table_count(A, d):
    """
    Using a dynamic table (see thesis), this function returns whether prefix-disc is smaller than d and 
    how many colorings lead to this prefix discrepancy.

    :param A: incidence matrix
    :type A: numpy.ndarray
    :param d: constant d used in dynamic program
    :return: boolean (true if prefix-disc(A) <= d) + count (# of colorings)
    """
    m, n = A.shape
    D_1 = np.empty(((2*d)+1)**m)
    D_2 = np.empty(((2*d)+1)**m)
    permutations = cartesian_jit(m, d)
    
    # Build table D[][] in bottom up manner
    for i in range(n + 1):
        for v in permutations:
            v_i = v_to_ind(d, v)
            if (i==0):
                if (np.count_nonzero(v)==0):
                    D_2[v_i] = 1
                else:
                    D_2[v_i] = 0
            else:
                min = 0
                plus = 0
                if (bounded(v - A[:,i-1], d) == True):
                    min = D_1[v_to_ind(d, v - A[:,i-1])]
                if (bounded(v + A[:,i-1], d) == True):
                    plus = D_1[v_to_ind(d, v + A[:,i-1])]
                D_2[v_i] = min + plus
        D_1 = D_2
        D_2 = np.empty(((2*d)+1)**m)
        
    count = 0
    for v in permutations:
        v_i = v_to_ind(d, v)
        count += D_1[v_i]
    
    boolean = False
    if count > 0:
        boolean = True

    return boolean, count

@njit(cache=True)
def calc_prefix_disc_dp(incidence):
    """
    Function that augments d progressively and returns the first d for which dynamic_table(A,d) is true.

    :param incidence: incidence matrix
    :type incidence: numpy.ndarray
    :return: prefix-disc (int) of incidence matrix
    """
    prefix_disc = -1
    m, n = incidence.shape
    for d in range(2*m):
        if(dynamic_table(incidence, d) == True):
            prefix_disc = d
            break
    return prefix_disc

@njit(cache=True)
def calc_prefix_disc_dp_count(incidence):
    """
    Function that augments d progressively and returns the first d for which dynamic_table(A,d) is true.
    Also, this function returns the number of colorings that lead to this particular prefix-disc.

    :param incidence: incidence matrix
    :type incidence: numpy.ndarray
    :return: prefix-disc (int) of incidence matrix + # of colorings
    """
    prefix_disc = -1
    count = -1
    m, n = incidence.shape
    for d in range(2*m):
        boolean, count = dynamic_table_count(incidence, d)
        if(boolean == True):
            prefix_disc = d
            break
    return prefix_disc, count