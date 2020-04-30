
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import pandas as pd


#构建一个稀疏表示分类器
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

iris = pd.read_csv("test3.csv")

print(iris.shape)


x=iris[["mean","max","count","sex"]]
y=iris["weight"]

x_train,x_test,y_train,y_test=train_test_split(x.values,y.values,test_size=0.1)



ss = StandardScaler()

X_train = ss.fit_transform(x_train)
X_test = ss.transform(x_test)



knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
y_predict = knc.predict(X_test)

print('K近邻分类器的精确度为: ', knc.score(X_test, y_test))


#x_train是一个矩阵
print(x_train)
print(y_train)
#	输入测试样本的特征矩阵M=[x_1…x_i…x_k]。
#	将该特征矩阵归一化处理。
#	通过上述公式计算残差y^*=arg⁡〖min⁡(e_i (y),e_i (y)=||y-X_i*k_i ||)〗。
#	计算差值，进行分类处理。

minVals=x_train.min(0)
maxVals=x_train.max(0)
ranges = maxVals - minVals

m = x_train.shape[0]
normDataSet = np.zeros(np.shape(x_train))       #  np.shape(group) 返回一个和group一样大小的数组，但元素都为0
diffnormData =x_train - np.tile(minVals,(m,1))  #  (oldValue-min)  减去最小值
normDataSet1 =diffnormData / np.tile(ranges,(m,1))
print(minVals)
print(maxVals)
print(normDataSet1)


y=(y_train,y_train,y_train,y_train)
#k为稀疏系数
endmatrix=normDataSet1*y
print('endmatrix')
