# 引入必要的库函数
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import tf_export
import numpy as np
import gzip
import os
import struct

# 读取本地gz文档，并转换为numpy矩阵的函数
def load_localData():
    path = './'

    files = ['yelp.edgelist.gz']


    path=os.path.join(path,files[0])

    with gzip.open(path, 'rb') as f:
        s=f.read()
        s=s.replace(b'\n',b' ')
        s=s.split(b' ')
        s=s[:-1]
        s=[(int)(x.decode('ascii')) for x in s]

        A=np.array(s)
        A=A.reshape(((int)(A.shape[0]/3),3))
    print(A)
    return A



A=load_localData()
row=np.max(A[:,0])
col=np.max(A[:,1])
print(row,col)
B=np.zeros((row,col))
for a in A:
    B[a[0],a[1]]=a[2]
print(B)


