# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: M.Suryakumar
RegisterNumber:  212224040340
*/
```
```
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv("/content/Employee.csv")
print(data.head())
print(data.info())
data.isnull().sum()
print(data["left"].value_counts())
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
print(x.head())
y=data["left"]
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print(y_pred)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print(accuracy)
pre=dt.predict([[0.5,0.8,9,260,6,0,1,2]])
print(pre)

plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```
## Output:
![decision tree classifier model](sam.png)
![image](https://github.com/user-attachments/assets/fa030ba2-02b5-432a-b9b6-3513c7344f6a)
![image](https://github.com/user-attachments/assets/1a64aa11-2ea0-4735-a5ee-48f73cf603d6)
![image](https://github.com/user-attachments/assets/1b66feff-3e95-4109-b817-a38ffe272fa6)
![image](https://github.com/user-attachments/assets/c87a0836-109e-4a5d-a28b-7b1a89a0c19e)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
