import os
import sys
from sympy import *


def over(a,b,n):
    ans = 0.0
    p=symbols('p')
    now_p=1.0
    for i in range(n-1,-1,-1):
        #ans <<= 1
        ans += int(a ^ b == 0)*now_p
        now_p *= p if (b&1) == 1 else 1-p
        a &= (1 << i) - 1
        b >>= 1
    return ans

if __name__ == '__main__':
    n=int(sys.argv[1])
    print(over(0b001,0b100))
