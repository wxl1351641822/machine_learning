#找第k小的
import random
def Partition(A,m,p):
    v=A[m]
    i=m+1;
    # print("******************Partition******************","m=",m,",p=",p)
    while(True):

        while(i<=p and A[i]<v):
            # print("i=", i, A[i], v)
            i=i+1;
        while (p>=0 and A[p]>v):
            # print("p=",p,A[p],v)
            p=p-1;
        if i<p:
            # print("swap",i,p,"得到",A)
            temp=A[i];
            A[i]=A[p]
            A[p]=temp
        else:
            break;
    A[m]=A[p]
    A[p]=v

    # print("******************end_Partition******************","m=",m,",p=",p)
    return p;

def PartSelect(A,n,k):
    k=k-1
    m=0
    r=n
    A.append(10000)
    while(True):
        j = r
        j = Partition(A, m, j)
        if (k == j):
            return j
        elif (k < j):
            r=j
        else:
            m=j+1




A = list(range(0, 20))
random.shuffle(A)
j=PartSelect(A,len(A),5)
print(A[j])
print(A)