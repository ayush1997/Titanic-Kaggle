import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
import csv as csv
from itertools import izip

global ids
ids=[]
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
    df = df.drop(["PassengerId","Name","Ticket","Cabin"],axis=1)
    # print df
    df = np.array(df)



    return df

def passenger_id(file_name):
    df = pd.read_csv(file_name,delimiter=',',header=0)
    ids = df["PassengerId"].tolist()
    # print ids
    return ids

df = preprocessing("train.csv")
X_train = df[:,1:]
Y_train = df[:,0].tolist()


df = preprocessing("test.csv")
X_test = df
ids = passenger_id("test.csv")

pipeline=  Pipeline([
                    ('clf',DecisionTreeClassifier(criterion="entropy"))
                    ])

parameters={
    'clf__max_depth':(150,155,160),
    'clf__min_samples_split':(1,2,3),
    'clf__min_samples_leaf':(1,2,3)

}
# print Y_train[0]

#this is for without GridSearchCV
# titanic = DecisionTreeClassifier(criterion="entropy")
# titanic.fit(X_train,Y_train)

grid_search = GridSearchCV(pipeline,parameters,n_jobs=1,verbose=1,scoring='f1')

grid_search.fit(X_train,Y_train)

print "Best score:",grid_search.best_score_
print "Best parameters set:"
best_parameters = grid_search.best_estimator_.get_params()

for param_name in sorted(parameters.keys()):
    print (param_name,best_parameters[param_name])

# tree.export_graphviz(grid_search,out_file='tree.dot')

# prediction = t.predict(X_test)
prediction = grid_search.predict(X_test).tolist()
print prediction

with open("myfirstforest.csv", "wb") as predictions_file:
    # predictions_file = open("myfirstforest.csv", "wb")
    open_file_object = csv.writer(predictions_file, delimiter=',')
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(zip(ids,prediction))
    # predictions_file.close()
    print len(zip(ids,prediction))
