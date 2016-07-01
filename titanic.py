import pandas as pd
import numpy as np

df = pd.read_csv('train.csv',delimiter=',',header=0)

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

print df[:10]
