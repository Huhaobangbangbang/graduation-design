#支持向量机svm的使用
from  sklearn import svm
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn import svm,metrics

foot_data=pd.read_csv("ALL.csv")


x=foot_data[["average","variance","standarddeviation","sex"]]
y=foot_data["weight"]

x_train,x_test,y_train,y_test=train_test_split(x.values,y.values,test_size=0.1)
clf=svm.SVC()
clf.fit(x_train,y_train)
pre=clf.predict(x_test)

score=metrics.accuracy_score(y_test,pre)
print(score)
