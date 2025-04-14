import numpy as np
import pandas as pd

df = pd.read_csv('/content/Forest Cover Type.csv')

df.head()

df.info()

df.describe()

df.isnull().sum()

df['Cover_Type'].value_counts()

x = df.drop('Cover_Type', axis=1)
y = df['Cover_Type']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train.shape

x_test.shape

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lg = LogisticRegression()
lg = lg.fit(x_train, y_train)
y_pred = lg.predict(x_test)

accuracy_score(y_test, y_pred)

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc = dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)

accuracy_score(y_test, y_pred)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)

accuracy_score(y_test, y_pred)

df.iloc[100].values

user_input = (101, 2998,   45,    8,  351,   16, 5842,  223,  222,  134, 3721,
          1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0)

features = np.array(user_input)

rfc.predict(features.reshape(1,-1))

import pickle
pickle.dump(rfc, open('rfc.pkl','wb'))

df.iloc[10].values
