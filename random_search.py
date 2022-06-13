import numpy as np
from numba import njit
from hypergraph_njit import calc_prefix_disc_simple

M = 7
N = 7

print("running...")

for i in range(100):

    found = False
    counter = 1

    while(found==False):
        
        rnd_matrix = np.random.randint(2, size=(M,N))
        prefix_disc, count  = calc_prefix_disc_simple(rnd_matrix)
        
        if prefix_disc >= 3:
            with open('search_count.txt', 'a') as f:
                f.write(str(counter)+"\n")
            with open('species.txt', 'a') as f:
                f.write(str(rnd_matrix)+"\n")

            found = True

        counter += 1

print("done")