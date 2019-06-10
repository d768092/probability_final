import sys
import numpy as np

def over(A,B):
    a = bin(A)[2:].zfill(n)
    b = bin(B)[2:].zfill(n)
    ans = 0
    for i in range(n,0,-1):
        ans *= 2
        if a[-i:]==b[:i]:
            ans += 1
    return ans

if __name__ == '__main__':
    n = int(sys.argv[1])
    total = 1<<n
    rate = np.zeros((total,total))
    for i in range(total):
        for j in range(total):
            if i != j:
                oddB = over(i,i)-over(i,j)
                oddA = over(j,j)-over(j,i)
                rate[i][j] = round(oddA/(oddA+oddB),3)

    print(rate)