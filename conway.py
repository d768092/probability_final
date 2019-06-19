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

@jit(nopython=True)
def init(n):
    total = 1 << n
    rate = np.full((total, total),0.5)
    for i in range(total):
        for j in range(total):
            if i != j:
                BB, BA, AB, AA = over(j, j,n), over(j, i,n), over(i, j,n), over(i, i,n)
                rate[i,j]=(BB - BA) / (BB - BA - AB + AA)
    #rate = np.array([init_multi(i, j, n) for i in range(total) for j in range(total)]).reshape(total, total)
    return rate

def multi(rate, n, i):
    total = 1 << n
    subpro=rate[[i^(1<<j) for j in range(n)]+[i]]

    A_choices = np.argmax(subpro, axis=0)
    '''
    if i == 4:
        print([i^(1<<j) for j in range(n)]+[i],subpro,sep='\n')#print(list(map(bin,list(i^(1<<A_choices))+[i])))'''
    subpro = np.max(subpro,axis=0)
    '''if i==4:
        print(subpro)'''
    B_choice = np.argmin(subpro)
    A_choice2 = i^(1<<A_choices[B_choice]) if A_choices[B_choice]<n else i
    # A, B, A, rate
    return i, B_choice, A_choice2, subpro[B_choice]


def change1bit(rate, n):
    total = 1<<n
    return np.array(Parallel(n_jobs=os.cpu_count(), backend="threading")(delayed(multi)(rate,n,i) for i in range(total)))   
    
        

def main(n):
    '''total = 1<<n
    rate = np.zeros((total,total))
    for i in range(total):
        for j in range(total):
            if i != j:
                oddB = over(i,i)-over(i,j)
                oddA = over(j,j)-over(j,i)
                rate[i][j] = round(oddA/(oddA+oddB),3)

    print(rate) '''
    total = 1<<n
    rate = init(n)

    '''
    print('basic:')
    max_val_index = np.min(rate,axis=1)==np.max(np.min(rate,axis=1))
    for i in np.arange(rate.shape[0])[max_val_index]:
        min_index = np.argmin(rate[i])
        print('A=' + bin(i)[2:].zfill(n), 'B=' + bin(min_index)[2:].zfill(n), 'rate=' + str(rate[i,min_index]))
    for i in np.arange(rate.shape[0])[~max_val_index]:
        min_index = np.argmin(rate[i])
        print('A=' + bin(i)[2:].zfill(n), 'B=' + bin(min_index)[2:].zfill(n), 'rate=' + str(rate[i,min_index]))
    '''
    '''
    print(' '*n, *[' %*s'%(max(n,5), bin(i)[2:].zfill(n)) for i in range(total)])
    for i,row in enumerate(rate):
        print(bin(i)[2:].zfill(n),end=' ')
        for item in row:
            print(' %*.3f' % (max(n,5),item), end=' ')
        print()
    '''
    print('change 1 bit:')
    ch1 = change1bit(rate, n)
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
