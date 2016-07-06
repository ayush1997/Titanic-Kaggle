import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
import csv as csv
from itertools import izip
from scipy.stats import pointbiserialr, spearmanr
import matplotlib.pyplot as plt

global ids
ids=[]

def parameter_grid(param_df,df):
    scoresCV = []
    scores = []
    for j in range(1):
        scoresCV = []
        scores = []
        for i in range(1,len(param_df)):
            new_df=df[param_df.index[0:i+1].values]
            X = new_df.ix[:,1::]
            y = new_df.ix[:,0]
            clf = LogisticRegression()
            scoreCV = cross_validation.cross_val_score(clf, X, y, cv=3)
            print new_df.head()
            print np.mean(scoreCV)
            scores.append(np.mean(scoreCV))

        plt.figure(figsize=(15,5))
        plt.plot(range(1,len(scores)+1),scores, '.-')
        plt.axis("tight")
        plt.title('Feature Selection', fontsize=14)
        plt.xlabel('# Features', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.show()

def correlation(df):
    columns = df.columns.values
    print columns
    param=[]
    correlation=[]
    abs_corr=[]
    covariance = []

    # cor = np.array(df)
    # # print cor
    # x = cor[:,1:]
    # # print x
    # y = np.array(cor[:,0])
    # # print y
    # X = np.vstack((y,x))
    # print np.cov(X)

    for c in columns:
        #Check if binary or continuous
        if len(df[c].unique())<=2:
            corr = spearmanr(df['Survived'],df[c])[0]
            y = df['Survived']
            x = df[c]
            X = np.vstack((y,x))
            covar = np.cov(X)
        else:
            corr = pointbiserialr(df['Survived'],df[c])[0]
            print corr
            y = df['Survived']
            x = df[c]
            X = np.vstack((y,x))
            covar = np.cov(X)
        param.append(c)
        correlation.append(corr)
        abs_corr.append(abs(corr))
        covariance.append(covar[0][1])
    print covariance

    #Create dataframe for visualization
    param_df=pd.DataFrame({'correlation':correlation,'parameter':param, 'abs_corr':abs_corr,'covariance':covariance})

    #Sort by absolute correlation
    param_df=param_df.sort_values(by=['abs_corr'], ascending=False)

    #Set parameter name as index
    param_df=param_df.set_index('parameter')

    parameter_grid(param_df,df)

    print param_df

def preprocessing(file_name):
    df = pd.read_csv(file_name,delimiter=',',header=0)


    # print df.head()
    print df.columns.values
    # print df.dtypes

    #filled empty ages with their median
    df['Age'] = df['Age'].fillna(df["Age"].median())
    df['Fare'] = df['Fare'].fillna(df["Fare"].median())
    # print df.describe()['Age']

    #filled male with 0 and female with 1
    df.loc[df["Sex"]=="male","Sex"] = 0
    df.loc[df["Sex"]=="female","Sex"] = 1
    # print df[df['Sex']==1].count()
    # print df["Sex"]

    # print df["Embarked"].unique()
    # print df.loc[df["Embarked"]=="nan","Embarked"].describe()

    df["Embarked"] = df["Embarked"].fillna("S")
    df.loc[df["Embarked"]=="S","Embarked"]=0
    df.loc[df["Embarked"]=="C","Embarked"]=1
    df.loc[df["Embarked"]=="Q","Embarked"]=2

    #dropped uneccesary variables
    df = df.drop(['PassengerId','Name','Ticket','Cabin',"Embarked"],axis=1)
    # ['PassengerId' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch' 'Ticket' 'Fare','Cabin' 'Embarked']

    print df.head()

    if file_name == "train.csv":
        correlation(df)

    df = np.array(df)



    return df

def passenger_id(file_name):
    df = pd.read_csv(file_name,delimiter=',',header=0)
    ids = df["PassengerId"].tolist()
    # print ids
    return ids

def make_grid(clf):
    tree.export_graphviz(clf,out_file='tree.dot')


df = preprocessing("train.csv")
X_train = df[:,1:]
Y_train = df[:,0].tolist()


df = preprocessing("test.csv")
X_test = df
ids = passenger_id("test.csv")

pipeline=  Pipeline([
                    # ('clf',DecisionTreeClassifier(criterion="entropy"))
                    # ('clf',RandomForestClassifier())
                    ('clf',LogisticRegression())
                    ])

parameters={
    # 'clf__n_estimators':(100,70,50),
    # 'clf__max_depth':(150,155,160),
    # 'clf__min_samples_split':(1,2,3),
    # 'clf__min_samples_leaf':(1,2,3)

}
# print Y_train[0]

#this is for without GridSearchCV
# titanic = DecisionTreeClassifier(criterion="entropy")
# titanic.fit(X_train,Y_train)

grid_search = GridSearchCV(pipeline,parameters,n_jobs=1,verbose=1,scoring=None)

grid_search.fit(X_train,Y_train)

print "Best score:",grid_search.best_score_
print "Best parameters set:"
best_parameters = grid_search.best_estimator_.get_params()

for param_name in sorted(parameters.keys()):
    print (param_name,best_parameters[param_name])



# prediction = t.predict(X_test)
prediction = grid_search.predict(X_test).tolist()
# print prediction

with open("logistic.csv", "wb") as predictions_file:
    # predictions_file = open("myfirstforest.csv", "wb")
    open_file_object = csv.writer(predictions_file, delimiter=',')
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(zip(ids,prediction))
    # predictions_file.close()
    print "Done"
