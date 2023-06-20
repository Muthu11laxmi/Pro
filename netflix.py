#4.APPENDIX
#4.1 SOURCE CODE:
#IMPORTING PACKAGE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model                                            
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#READING FILE
import pandas as pd
df=pd.read_csv("NFLX.csv")

#DATA
print(df)


#HEAD OF THE DATASET

print("Head of the dataset")
print(df.head())

#TAIL OF THE DATASET

print("Tail of the dataset")
print(df.tail())

#SHAPE OF THE DATASET

print("Shape of the dataset")
print(df.shape)
#INFORMATION OF THE DATASET
print("Information of the dataset")
print(df.info())

#FINDING THE COUNT OF THE VALUES

print("Count the values:",df.count())

#FINDING THE MISSING VALUES IN THE DATASET

print("missing values in the dataset:")
print(df.isnull().sum())


#FINDING DUPLICATES ENTRIES
d=df[df.duplicated()]
print("Duplicate entries:")
print(d)

#DESCRIPTIVE STATISTICS

print("Mean=\n",df.mean())
print("Median=\n",df.median())
print("Variance=\n",df.var())
print("Standard deviation=\n",df.std())
print("Maximum value=\n",df.max())
print("Minimum value=\n",df.min())
#INTERQUANTILE
print("Interquartile=",df.quantile())
#AGGREGATE FUNCTION 
x=df.aggregate(["sum"])
print(x)
y=df.aggregate(["max"])
print(y)
z=df.aggregate(["mean"])
print(z)
s=df.aggregate(["sem"])
print(s)
p=df.aggregate(["var"])
print(p)
q=df.aggregate(["prod"])
print(q)
#DESCRIPTIVE STATISTICS FOR GROUPED DATA
df1=df.groupby(['High'])
print(df1.first())
print("Mean=\n",df1['High'].mean())
print("Median=\n",df1['High'].median())
print("Variance=\n",df1['High'].var())
print("Standard deviation=\n",df1['High'].std())
print("Maximum value=\n",df1['High'].max())
print("Minimum value=\n",df1['High'].min())
#SKEWNESS
print(df.skew())
#KURTOSIS
print(df.kurtosis())
#VISUALIZATION 
import seaborn as sns
fig=plt.figure()
ax=plt.axes(projection='3d')
x=df['Open']
y=df['Close']
z=df['High']
ax.plot3D(x,y,z,'purple')
ax.set_title('netflix stock prediction dataset')
plt.show()
plt.plot(df.High,df.Low)
plt.title("High vs Low ")
plt.xlabel("High")
plt.ylabel("Low ")
plt.show()
sns.pairplot(data=df)
plt.show()
plt.hist(df.High,bins=30)
plt.title("High")
plt.xlabel("High")
plt.show()
f,ax=plt.subplots(figsize=(10,6))
x=df['High']
ax=sns.kdeplot(x,shade=True,color='r')
plt.show()
sns.heatmap(df.corr())
plt.show()
#LINEAR REGGRESSION
x=df['High']
y =df['Low']
x.head()
y.head()
x.shape
y.shape
x=x.values.reshape(-1,1)
x.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,test_size=0.1,random_state=100)
print(x.shape)
print(y.shape)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)
regr.coef_
regr.intercept_plt.scatter(x_train, y_train)
plt.plot(x_train,-5.275 + 0.9820*x_train, 'r')
plt.show()
y_pred = regr.predict(x_test)
r_squared = r2_score(y_test, y_pred)
r_squared
print('Mean squared error: %.2f'% mean_squared_error(y_test, y_pred))
print('Mean Absolute Error: %.2f'% mean_absolute_error(y_test, y_pred))

#MULTIPLE REGRESSION
X=df[['High','Low','Open']]
Y=df['Close']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=101)
reg=linear_model.LinearRegression()
reg.fit(X_train,Y_train)Y_predict=reg.predict(X_test)
print('Coefficients:',reg.coef_)
print('Variance score:{}'.format(reg.score(X_test,Y_test)))
from sklearn.metrics import r2_score
print('r^2:',r2_score(Y_test,Y_predict)) 
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y_test,Y_predict)
rmse=np.sqrt(mse)
print('RMSE:',rmse)



