#找第k小的,优化了点
import random
import math
def Swap(a,b):
    return b,a

def InSort(A,m,p):
    for i in range(m,p):
        max=m
        for j in range(m,p-i+1):
            if(A[j]>A[max]):
                max=j
        A[p-i],A[max]=Swap(A[p-i],A[max])

def Partition(A,m,p):
    v=A[m]
    i=m+1;
    # print("******************Partition******************","m=",m,",p=",p)
    while(True):

        while(i<=p and A[i]<v):
            # print("i=", i, A[i], v)
            i=i+1;
        # print(p)
        while (p>=0 and A[p]>v):
            # print("p=",p,A[p],v)
            p=p-1;

        if i<p:
            # print("swap",i,p,"得到",A)
            A[p],A[i]=Swap(A[p],A[i])
        else:
            break;
    A[m]=A[p]
    A[p]=v

    # print("******************end_Partition******************","m=",m,",p=",p)
    return p;

def Select(A,m,p,k):
    r=5
    if p-m+1<=r:
        InSort(A,m,p)
        return m+k-1
    while(True):
        n=p-m+1
        for i in range(1,int(n/r)):
            InSort(A,m+(i-1)*r,m+r*i-1)
            A[m+i-1],A[m+(i-1)*r+(int)(r/2)-1]=Swap(A[m+i-1],A[m+(i-1)*r+(int)(r/2)-1])
        j=Select(A,m,m+int(n/r)-1,math.ceil(int(n/r)/2))
        A[m],A[j]=Swap(A[m],A[j])
        j=p
        j=Partition(A,m,j)
        if(j-m+1==k):
            return j;
        elif(j-m+1>k):
            p=j-1
        else:
            m=j+1

A = list(range(0, 20))
random.shuffle(A)
j=Select(A,0,len(A)-1,2)
print(A[j])
print(A)