import os
import sys
import numpy as np
from numba import jit, prange
import numba as nb
from joblib import Parallel, delayed
from multiprocessing import Pool

@jit(nopython=True)
def over(a,b,n):
    ans = 0
    for i in range(n-1,-1,-1):
        ans <<= 1
        ans += int(a ^ b == 0)
        a &= (1 << i) - 1
        b >>= 1
    return ans

def multi(rate, n, i):
    total = 1 << n
    subpro=rate[[i^(1<<j) for j in range(n)]+[i]]

    A_choices = np.argmax(subpro, axis=0)
    subpro = np.max(subpro,axis=0)
    B_choice = np.argmin(subpro)
    A_choice2 = i^(1<<A_choices[B_choice]) if A_choices[B_choice]<n else i
    # A, B, A, rate
    return i, B_choice, A_choice2, subpro[B_choice]


def change1bit(rate, n):
    total = 1<<n
    return np.array(Parallel(n_jobs=os.cpu_count(), backend="threading")(delayed(multi)(rate,n,i) for i in range(total)))   
    
@jit(nopython=True)
def multi_less_mem(tup):
    n, i = tup
    total = 1<<n
    
    #generate subpro
    rate = 100
    for j in range(total):
        subpro_col=np.empty(n+1, dtype=np.float64)
        for a in range(n+1):
            actually_a=i^(1<<a) if a<n else i
            if actually_a == j:
                subpro_col[a]=0.5
                continue
            BB, BA, AB, AA = over(j, j, n), over(j, actually_a, n), over(actually_a, j, n), over(actually_a, actually_a, n)
            subpro_col[a]=(BB - BA) / (BB - BA - AB + AA)
        max_choice = np.argmax(subpro_col)
        if subpro_col[max_choice] < rate:
            B_choice = j
            A_choice_idx = max_choice
            rate = subpro_col[max_choice]

    A_choice = i^(1<<A_choice_idx) if A_choice_idx<n else i
    # A, B, A, rate
    return float(i), float(B_choice), float(A_choice), rate

def change1bit_less_mem(n):
    total = 1 << n
    multi_less_mem((3,0))
    with Pool(processes=os.cpu_count()) as poo:
        return np.array(poo.map(multi_less_mem,[(n, i) for i in range(total)]))
        

def main(n):
    total = 1<<n
    print('change 1 bit:')
    ch1 = change1bit_less_mem(n)
    max_index = np.argmax(ch1[:, 3])
    
    max_index = np.max(ch1[:, 3])
    for item in ch1[ch1[:, 3] == max_index]:
        print('A=',bin(int(item[0]))[2:].zfill(n),' B=',bin(int(item[1]))[2:].zfill(n),' A=',bin(int(item[2]))[2:].zfill(n),' rate=',item[3],sep='')
    
    '''
    for item in ch1[ch1[:, 3] != max_index]:
        print('A=',bin(int(item[0]))[2:].zfill(n),' B=',bin(int(item[1]))[2:].zfill(n),' A=',bin(int(item[2]))[2:].zfill(n),' rate=',item[3],sep='')
    '''


if __name__ == "__main__":
    n = int(sys.argv[1])
    main(n)
