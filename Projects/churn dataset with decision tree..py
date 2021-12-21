import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings(action = 'ignore')

data = pd.read_csv('churn_dataset.csv')
data.info()

data.head()

XX = data.drop(columns = ['customer_id','churn'])
Y = data['churn']
X = XX


X.head()

#preprocessing the data
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X = scale.fit_transform(X)

from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test = train_test_split(X, Y, train_size = 0.8)
x_train.shape, y_train.shape, x_test.shape

#creating model
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(class_weight = 'balanced')

classifier.fit(x_train,y_train)
pred_values = classifier.predict(x_test)
pred_values

from sklearn.metrics import classification_report
print(classification_report(y_test,pred_values))

classifier = DecisionTreeClassifier(max_depth = 9)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print(classification_report(y_test,y_pred))

#plotting the important features in the dataset
features_imp = pd.Series(classifier.feature_importances_, index = XX.columns)
k = features_imp.sort_values()
plt.figure(figsize = (5,4))
plt.barh(k.index ,k)
