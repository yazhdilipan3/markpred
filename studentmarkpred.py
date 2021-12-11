import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
s = pd.read_csv("mark.csv",sep = r',',names=["name","fedu","medu","g1","g2","g3"])
#s= s["g2"].describe()
"""
demo = sns.countplot(s['g2'])
demo.axis.set_title("DISTRIBUTION",fontsize = 35)
demo.set_xlabel("final grade")
demo.set_ylabel("count")
plt.show()
"""
m = s
print(m.columns)
from sklearn.model_selection import train_test_split

x = m.drop("g2",axis = 1)
y = m["g2"]
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 44)
from sklearn.linear_model import LinearRegression
l = LinearRegression()
l.fit(X_train,y_train)
y_pred = l.predict(X_test)
print(y_pred)