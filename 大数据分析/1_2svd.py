import numpy as np


def svd(A):
    ATA=np.dot(A.T,A)
    AAT=np.dot(A,A.T)
    ATA_vals, ATA_vecs = np.linalg.eig(ATA)#A=P*B*PT,故使用np中的eig分解出的vecs的列向量是特征向量。A = P*B*P-1

    # ATA_vals=np.diag(ATA_vals)
    # print(ATA.shape,ATA_vals.shape,ATA_vecs.shape)
    print(np.allclose(ATA,np.dot(ATA_vecs,np.dot(ATA_vals,ATA_vecs.T))))
    AAT_vals, AAT_vecs = np.linalg.eig(AAT)
    # AAT_vals = np.diag(AAT_vals)
    print(ATA_vals)
    print(AAT_vals)
    smat=np.sqrt(AAT_vals)
    return AAT_vecs,smat,ATA_vecs.T

A=np.array([np.array([1,5,7,6,1]),np.array([2,1,10,4,4]),np.array([3,6,7,5,2])])

print(A)
U,smat,VT=svd(A)
u, s, vh = np.linalg.svd(A)
print('*******************U****************************')
print(U)
print(u)
print('*******************s****************************')
print(smat)
print(s)
print('*******************vh****************************')
print(VT)
print(vh)
ss=np.zeros((3,5))
ss[:3, :3] = np.diag(smat)
print('*******************A****************************')
print(np.dot(U, np.dot(ss, VT)))
print(np.allclose(A, np.dot(U, np.dot(ss, VT))))
ss[:3, :3] = np.diag(s)
print(np.dot(u, np.dot(ss, vh)))
print(np.allclose(A, np.dot(u, np.dot(ss, vh))))
# 因为UV中对应的顺序不同。
