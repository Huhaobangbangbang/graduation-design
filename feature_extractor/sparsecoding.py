
from time import time

#读取Excel中的数据
import numpy as np
import xlrd
from  numpy import array
from  numpy import count_nonzero
from scipy.sparse import csr_matrix

data = xlrd.open_workbook('8.xlsm')
table = data.sheets()[0]

start=18#开始的行
end=55#结束的行
rows=end-start

count=0

list_values=[]
for x in range(start,end):
  values=[]
  row =table.row_values(x)
  for i in range(1,27):
   values.append(row[i])
  list_values.append(values)
#list_values就是我们要处理的原始矩阵

#array按列读取元素
#datamatrix=np.array(list_values)
#进行稀疏提取特征
#整张头像的照片是个图片，通过对图片的分割可以将图片分割为一个一个的小块，也就是一个个Pitch。

S=csr_matrix(list_values)
#print(S)

B=S.todense()
print(B)


#
A=array(list_values)

sparsity=1.0-count_nonzero(list_values)/A.size
print(sparsity)
