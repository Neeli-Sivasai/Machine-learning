import pandas as pd
import matplotlib.pyplot as plt

#reading data
data = pd.read_csv('Iris.csv')
#null values
data.isnull().sum()

#counting elements
data['Species'].value_counts()

#analysing data by plotting with matplotlib
data.groupby('Species')['PetalWidthCm'].mean().plot(kind='bar')
plt.bar(data['Species'].unique(), data['Species'].value_counts())

#data cleaning
data.drop(columns='PetalWidthCm')

#dividing the features
y = data['Species']
x = data.iloc[:,0:4]

from sklearn.model_selection import train_test_split

# Splitting the Dataset 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.33)

#model preaparing
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)

#prediction
predicted_values = neigh.predict(x_test)
predicted_values

#calculating accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predicted_values)

#wow! i got 100% accuracy, may be due to the dataset was too small.... and we selected the perfect algorithm to train and predicting
