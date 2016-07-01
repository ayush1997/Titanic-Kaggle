import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

def preprocessing(file_name):
    df = pd.read_csv(file_name,delimiter=',',header=0)


    # print df.head()
    # print df.columns
    # print df.dtypes

    #filled empty ages with their median
    df['Age'] = df['Age'].fillna(df["Age"].median())
    df['Fare'] = df['Fare'].fillna(df["Fare"].median())
    # print df.describe()['Age']

    #filled male with 0 and female with 1
    df.loc[df["Sex"]=="male","Sex"] = 0
    df.loc[df["Sex"]=="female","Sex"] = 1
    print df[df['Sex']==1].count()
    # print df["Sex"]

    # print df["Embarked"].unique()
    # print df.loc[df["Embarked"]=="nan","Embarked"].describe()

    df["Embarked"] = df["Embarked"].fillna("S")
    df.loc[df["Embarked"]=="S","Embarked"]=0
    df.loc[df["Embarked"]=="C","Embarked"]=1
    df.loc[df["Embarked"]=="Q","Embarked"]=2

    #dropped uneccesary variables
    df = df.drop(["Fare","Age","PassengerId","Name","Ticket","Cabin"],axis=1)
    print df
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


titanic = DecisionTreeClassifier(max_depth=150)
titanic.fit(X_train,Y_train)

tree.export_graphviz(titanic,out_file='tree.dot')
prediction = titanic.predict(X_test)


print prediction

# print titanic.score(prediction,Y_test)
