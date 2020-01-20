# A=[1,9,5,3,7,8,2,35]*10
import random
A=list(range(0,101))
random.shuffle(A)
n=len(A)
def Partition(m,p):
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
            print("swap",i,p,"得到",A)
            temp=A[i];
            A[i]=A[p]
            A[p]=temp
        else:
            break;
    A[m]=A[p]
    A[p]=v

    # print("******************end_Partition******************","m=",m,",p=",p)
    return p;

def QuickSort(p,q):
    if p<q:
        # print("******************QuickSort******************", "p=", p, ",q=", q)
        j=Partition(p,q);
        print("划分，p=", p,"j=",j, ",q=", q,"A=",A)
        # print("left:")
        QuickSort(p,j-1);
        # print("right:")
        QuickSort(j+1,q);
        # print("******************end_QuickSort******************", "p=", p, ",q=", q)
print(A)
QuickSort(0,n-1)
print(A)
