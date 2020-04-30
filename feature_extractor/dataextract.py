
#读取Excel中的数据
import numpy as np
import xlrd
import xlwt
import os
import xlsxwriter


data = xlrd.open_workbook('16.xlsm')
table = data.sheets()[0]

start=18#开始的行
end=50#结束的行
rows=end-start



list_values=[]
for x in range(start,end):
  values=[]
  row =table.row_values(x)
  a=1
  b=24
  for i in range(a,b):
   values.append(row[i])
  list_values.append(values)
 # print(list_values)
count=rows*(b-a)

#array按列读取元素
datamatrix=np.array(list_values)
mean=np.mean(datamatrix)#计算平均值
var=np.var(datamatrix)#计算方差
std=np.std(datamatrix)#计算标准差
max=np.max(datamatrix)  #计算数字矩阵中的最大值


#保留一位小数
mean1=round(mean,1)
var1=round(var,1)
std1=round(std,1)
max1=round(max,1)




print(datamatrix)
print(mean)
print( var)
print(std)
print(max)
print(count)
#  将数据写入新文件
a = os.getcwd() #获取当前目录
print (a) #打印当前目录
#创建一个workbook对象，相当于创建一个Excel文件
workbook = xlsxwriter.Workbook("shit.xlsx")

# 创建一个sheet对象，一个sheet对象对应Excel文件中的一张表格。

worksheet = workbook.add_worksheet()
# 向表中添加数据标题
worksheet.write('A1', mean1)
worksheet.write('B1', var1)
worksheet.write('C1', std1)
worksheet.write('D1',max1)
worksheet.write('E1',count)
workbook.close()
