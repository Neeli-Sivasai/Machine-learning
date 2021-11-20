import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing data from local disc
data = pd.read_csv('diabetes.csv')

#to check any null values in the dataset
data.isnull().sum()

#to understand the data briefly
data.describe()
data.info()

#shape of the data
data.shape
data.head(4)

#spilliting training(x) and target variables(y)
y = data.iloc[:,8]
x = data.iloc[:,0:8]

#data preprocessing
from sklearn import preprocessing
scale = preprocessing.StandardScaler()
columns = x.columns
x = scale.fit_transform(x)
x = pd.DataFrame(x, columns = columns)
x.head()

from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test = train_test_split(x,y, test_size = 0.2)

x.shape, x_train.shape, x_test.shape, y_test.shape
y.value_counts()

#Creating model
#DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import cross_val_score as cvs
score = cvs(DTC(),x,y, cv=5)
score
score.mean()

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier as RFC
score = cvs(RFC(),x,y, cv = 5)
score.mean()

#BaggingClassifier
from sklearn.ensemble import BaggingClassifier
bag_model = BaggingClassifier(base_estimator = DTC(),
                              n_estimators = 100,
                              max_samples = 0.8)

score = cvs(bag_model,x,y, cv=5)
score.mean()
