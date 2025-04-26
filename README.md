# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Naveen Kumar V
RegisterNumber:  212223220068
*/
```
```

import pandas as pd
data=pd.read_csv("Salary.csv")
print(data.head())

print(data.info)

print(data.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
print(data.head())

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

```
## Output:
## data.head()
![Screenshot 2025-04-26 133648](https://github.com/user-attachments/assets/304b4743-f161-440b-b187-5e1fd7370cbb)
## data.isnull().sum()
![Screenshot 2025-04-26 133703](https://github.com/user-attachments/assets/c09a8315-065a-4a94-8c76-b777793d199d)
## data.head() for salary
![Screenshot 2025-04-26 133708](https://github.com/user-attachments/assets/e2c00d4e-e730-493e-a580-78dbccf2ffe4)
## MSE and r2 value
![Screenshot 2025-04-26 134150](https://github.com/user-attachments/assets/7514c1a9-a777-4e98-ad6f-78a984f4ffa7)
## data prediction
![Screenshot 2025-04-26 133713](https://github.com/user-attachments/assets/712f3328-1a21-4be8-b7c5-538b46b89fbc)
## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
