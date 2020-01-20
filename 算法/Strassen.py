import numpy as np
A=np.random.rand(4,4)
B=np.random.rand(4,4)


def Strassen_(A,B):
    size = A.shape

    A11 = A[0:int(size[0] / 2), 0:int(size[1] / 2)]
    A12 = A[0:int(size[0] / 2), int(size[1] / 2):size[1]]
    A21 = A[int(size[0] / 2):size[0], 0:int(size[1] / 2)]
    A22 = A[int(size[0] / 2):size[0], int(size[1] / 2):size[1]]
    size = B.shape
    B11 = B[0:int(size[0] / 2), 0:int(size[1] / 2)]
    B12 = B[0:int(size[0] / 2), int(size[1] / 2):size[1]]
    B21 = B[int(size[0] / 2):size[0], 0:int(size[1] / 2)]
    B22 = B[int(size[0] / 2):size[0], int(size[1] / 2):size[1]]

    P=np.dot(np.add(A11,A22),np.add(B11,B22))
    Q=np.dot(np.add(A21,A22),B11)
    R=np.dot(A11,np.subtract(B12,B22))
    S = np.dot(A22, np.subtract(B21, B11))
    T=np.dot(np.add(A11,A12),B22)
    U=np.dot(np.subtract(A21,A11),np.add(B11,B12))
    V=np.dot(np.subtract(A12,A22),np.add(B21,B22))

    C11=np.array(np.add(np.subtract(np.add(P,S),T),V))
    C12=np.array(np.add(R,T))
    C21=np.add(Q,S)
    C22=np.add(np.subtract(np.add(P,R),Q),U)


    C1=np.c_[C11, C12]
    C2=np.c_[C21, C22]

    return np.r_[C1,C2]

print(Strassen_(A,B))
print(np.dot(A,B))