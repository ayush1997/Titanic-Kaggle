import pandas as pd

df = pd.read_csv('train.csv',delimiter=',',header=0)

print df.head()


df['Age'] = df['Age'].fillna(df["Age"].median())
print df.describe()['Age']
