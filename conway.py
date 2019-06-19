import os
import sys
import numpy as np
from numba import jit, prange
import numba as nb
from joblib import Parallel, delayed
from multiprocessing import Pool

@jit(nopython=True)
def over(a,b,n, p=0.5):
    ans = 0.0
    now_p=1.0
    for i in range(n-1,-1,-1):
        #ans <<= 1
        ans += int(a ^ b == 0)*now_p
        now_p *= p if (b&1) == 1 else 1-p
        a &= (1 << i) - 1
        b >>= 1
    return ans

@jit(nopython=True)
def init(n, p=0.5):
    total = 1 << n
    rate = np.full((total, total),1.0)
    for i in range(total):
        for j in range(total):
            if i != j:
                BB, BA, AB, AA = over(j,j,n,p), over(j,i,n,p), over(i,j,n,p), over(i,i,n,p)
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
    total = 1<<n
    
    st,ed,step=list(map(float, sys.argv[2:]))
    opt=(0,0,0,0)# p, i, min_index, rate[i,min_index] 
    for p in np.arange(st,ed+1e-7,step):
        rate = init(n,p)
        i=np.argmax(np.min(rate,axis=1))
        min_index = np.argmin(rate[i])
        print('p=%.5f, A=%s, B=%s, rate=%.10f' %(p, bin(i)[2:].zfill(n), bin(min_index)[2:].zfill(n), rate[i,min_index]))

        if rate[i,min_index] > opt[3]:
            opt=(p,i,min_index,rate[i,min_index])
    print('Optimal choice:\np=%.5f, A=%s, B=%s, rate=%.10f' %(opt[0], bin(opt[1])[2:].zfill(n), bin(opt[2])[2:].zfill(n), opt[3]))
    
    '''
    rate = init(n,0.5)
    print('basic:')
    max_val_index = np.min(rate,axis=1)==np.max(np.min(rate,axis=1))
    for i in np.arange(rate.shape[0])[max_val_index]:
        min_index = np.argmin(rate[i])
        print('A=' + bin(i)[2:].zfill(n), 'B=' + bin(min_index)[2:].zfill(n), 'rate=' + str(rate[i,min_index]))
    
    for i in np.arange(rate.shape[0])[~max_val_index]:
        min_index = np.argmin(rate[i])
        print('A=' + bin(i)[2:].zfill(n), 'B=' + bin(min_index)[2:].zfill(n), 'rate=' + str(rate[i,min_index]))

    print(' '*n, *[' %*s'%(max(n,5), bin(i)[2:].zfill(n)) for i in range(total)])
    for i,row in enumerate(rate):
        print(bin(i)[2:].zfill(n),end=' ')
        for item in row:
            print(' %*.3f' % (max(n,5),item), end=' ')
        print()
    '''
    '''
    print('change 1 bit:')
    ch1 = change1bit(rate, n)
    max_index = np.max(ch1[:, 3])
    for item in ch1[ch1[:, 3] == max_index]:
        print('A=',bin(int(item[0]))[2:].zfill(n),' B=',bin(int(item[1]))[2:].zfill(n),' A=',bin(int(item[2]))[2:].zfill(n),' rate=',item[3],sep='')'''
    
    '''
    for item in ch1[ch1[:, 3] != max_index]:
        print('A=',bin(int(item[0]))[2:].zfill(n),' B=',bin(int(item[1]))[2:].zfill(n),' A=',bin(int(item[2]))[2:].zfill(n),' rate=',item[3],sep='')
    '''


if __name__ == "__main__":
    n = int(sys.argv[1])
    main(n)
