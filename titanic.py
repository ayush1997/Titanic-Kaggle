import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

def preprocessing(file_name):
    df = pd.read_csv(file_name,delimiter=',',header=0)


    # print df.head()
    # print df.columns
    # print df.dtypes

    #filled empty ages with their median
    df['Age'] = df['Age'].fillna(df["Age"].median())

    # print df.describe()['Age']

    #filled male with 0 and female with 1
    df.loc[df["Sex"]=="male","Sex"] = 0
    df.loc[df["Sex"]=="female","Sex"] = 1

    # print df["Sex"]

    # print df["Embarked"].unique()
    # print df.loc[df["Embarked"]=="nan","Embarked"].describe()

    df["Embarked"] = df["Embarked"].fillna("S")
    df.loc[df["Embarked"]=="S","Embarked"]=0
    df.loc[df["Embarked"]=="C","Embarked"]=1
    df.loc[df["Embarked"]=="Q","Embarked"]=2

    #dropped uneccesary variables
    df = df.drop(["PassengerId","Name","Ticket","Cabin"],axis=1)
    # print df.describe()
    # print df
    df = np.array(df)
    

    return df



df = preprocessing("train.csv")
X_train = df[:,1:]
Y_train = df[:,0]
print X_train[0]
# print Y_train
df = preprocessing("test.csv")
X_test = df
print X_test[152]

titanic = DecisionTreeRegressor()
titanic.fit(X_train,Y_train)

# for j,i in enumerate(X_test):
#     print j
#     prediction = titanic.predict(np.array(i))
#

# print prediction

# print titanic.score(prediction,Y_test)
