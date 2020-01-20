import time
import random
import copy
A=[]
B=[]
n=len(A)

def Merge(low,mid,high):
    h=low
    i=low
    j=mid+1

    while h<=mid and j<=high:
        if(A[h]<=A[j]):
            B[i]=A[h]
            h=h+1
        else:
            B[i]=A[j]
            j=j+1
        i=i+1
    if(h>mid):
        for k in range(j,high):
            B[i]=A[k]
            i=i+1
    else:
        for k in range(h, mid):
            B[i] = A[k]
            i = i + 1
    for k in range(low,high):
        A[k]=B[k]

def MergeSort(low,high):
    if low<high:
        mid=int((low+high)/2)
        MergeSort(low,mid)
        # print("左边的:low=",low,",mid=",mid,",A=",A)
        MergeSort(mid+1,high)
        # print("右边的:mid+1=",mid+1, ",high=",high,",A=", A)
        Merge(low,mid,high)
        # print("左右融合：low=",low,",mid=",mid,",high=",high,",A=", A)

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

def QuickSort(p,q):
    if p<q:
        # print("******************QuickSort******************", "p=", p, ",q=", q)
        j=Partition(p,q);
        # print("划分，p=", p,"j=",j, ",q=", q,"A=",A)
        # print("left:")
        QuickSort(p,j-1);
        # print("right:")
        QuickSort(j+1,q);


if __name__=='__main__':
    for i in range(1,11):
        A=list(range(0, 100 * i))
        # print(A)
        random.shuffle(A)
        # print(A)
        A1=copy.deepcopy(A)
        B=[0]*len(A)

        t0=time.time()
        # print("归并排序")
        MergeSort(0,len(A)-1)
        t1=time.time()

        A=copy.deepcopy(A1)
        n = len(A)
        t3=time.time()
        QuickSort(0,n-1)
        t2=time.time()
        # print(A)
        print("规模为"+str(len(A))+"的数组的排序，归并排序用时"+str(t1-t0)+",快速排序用时"+str(t2-t3))


