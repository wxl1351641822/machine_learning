#找第k小的,优化了点
import random
def InSort(A,m,p):
    for i in range(m,p):
        max=m
        for j in range(m,p-i+1):
            if(A[j]>A[max]):
                max=j
        temp=A[max]
        A[max]=A[p-i]
        A[p-i]=temp
        print(i,A)


A = list(range(0, 20))
random.shuffle(A)
print(A)
InSort(A,0,19)
print(A)