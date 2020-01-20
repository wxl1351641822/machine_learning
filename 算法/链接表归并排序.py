A=[1,4,5,6,7,8,2,35]
B = [0 for i in range(len(A))]
def InSort(A,B,low,high,p):
def MergeL(low,mid,high):
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

def MergeSortL(low,high,p):
    if low<high:
        mid=int((low+high)/2)
        if high-low+1<2:
            InSort(A,B,low,high,p)
        MergeSortL(low,mid)
        print("左边的:low=",low,",mid=",mid,",A=",A)
        MergeSortL(mid+1,high)
        print("右边的:mid+1=",mid+1, ",high=",high,",A=", A)
        MergeL(low,mid,high)
        print("左右融合：low=",low,",mid=",mid,",high=",high,",A=", A)

if __name__=='__main__':
    print("归并排序前：",A)
    MergeSort(0,len(A)-1)
    print("归并排序后：",A)

