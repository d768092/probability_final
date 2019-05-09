import numpy as np
import sys
n = int(sys.argv[1])
total = 1<<n
I = np.eye(total//2)
pro = np.zeros((total,total))
B = np.zeros((total//2,total//2))
for i in range(total//2):
    left = i%(total//4)
    B[i][2*left]=0.5
    B[i][2*left+1]=0.5
for i in range(total):
    for j in range(total):
        # x = [[p(i>j|00)],[p(i>j|01)],[p(i>j|10)],[p(i>j|11)]]
        # x = Ax + y
        # x = (inv(I-A))y
        A = B.copy()
        y = np.zeros((total//2,1))
        A[i>>1][i%(total>>1)]=0
        A[j>>1][j%(total>>1)]=0
        y[i>>1][0]=0.5
        pro[i][j]=np.linalg.inv(I-A).dot(y).squeeze().mean()

for i in range(total):
    for j in range(total):
        print('%.3f' % round(pro[i][j],3),end=' ')
    print('')
'''
for i in range(total):
    a1 = i>>1
    print('A='+str(i),'B='+str(a1),'pro='+str(pro[i][a1]))
    a2 = a1 + total//2
    print('A='+str(i),'B='+str(a2),'pro='+str(pro[i][a2]))
'''    
#print(np.round(pro,3))
print('basic:')
for i in range(total):
    min_index = np.argmin(pro[i])
    print('A='+str(i),'B='+str(min_index),'pro='+str(pro[i][min_index]))

print('change 1 bit:')
pro = pro - np.eye(total)
for i in range(total):
    subpro = pro.copy()
    choice=set([i^(1<<j) for j in range(n)])
    choice.add(i)
    for j in range(total):
        if j not in choice:
            subpro[j,:]*=0
    max_index=np.argmax(subpro,axis=0)
    subpro=np.max(subpro,axis=0)
    min_index = np.argmin(subpro)
    print('A='+str(i),'B='+str(min_index),'A='+str(max_index[min_index]),'pro='+str(subpro[min_index]))
