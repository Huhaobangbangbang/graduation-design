import  pandas as pd
import pandas_profiling

#分析数据库信息
#分析数据
data=pd.read_csv("data/d-1.csv")
pandas_profiling.ProfileReport(data)
profile = pandas_profiling.ProfileReport(data)
profile.to_file(outputfile = "output_file.html")

data.head()
