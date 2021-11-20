import pandas as pd 
import numpy as np                      
import seaborn as sns                   
import matplotlib.pyplot as plt  
%matplotlib inline 
import warnings    

data = pd.read_csv('Housing_Data.csv')
data.head()

#check the features present in our data
data.columns

# the shape of the dataset.
data.shape

data.describe()

# missing values
data.isnull().sum()

print("Before deleting outliers length = " , len(data))                
target = data['Sale_Price']
target_mean = target.mean()
target_sd = target.std()
data = data[(target > target_mean - 2*target_sd) & (target < target_mean + 2*target_sd)]
print("After deleting outliers length = " ,  len(data)) 

Numerical_Type = ['Sale_Price','Flat Area (in Sqft)', 'Lot Area (in Sqft)',
       'Area of the House from Basement (in Sqft)', 'Basement Area (in Sqft)',
       'Age of House (in Years)', 'Latitude', 'Longitude',
       'Living Area after Renovation (in Sqft)',
       'Lot Area after Renovation (in Sqft)']

Categorical_Type = []

for feature in data.columns:
    if (feature not in Numerical_Type):
        Categorical_Type.append(feature)

numerical_features   = data[Numerical_Type]
categorical_features = data[Categorical_Type]

numerical_features,categorical_features

def correlation_heatmap(data):
    _,  ax = plt.subplots(figsize = (25, 20))
    colormap= sns.diverging_palette(220, 10, as_cmap = True)
    sns.heatmap(data.corr(), annot=True, cmap = colormap)

correlation_heatmap(data)

#seperating independent and dependent variables

data_x = data.drop(['Sale_Price'], axis=1)
data_y = data['Sale_Price']

from sklearn.model_selection import train_test_split as tts

train1_x, test_x , train1_y, test_y = tts( data_x, data_y , test_size = 0.2 , random_state = 50)
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(data_x,data_y, random_state = 86)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
train_x = ss.fit_transform(train_x)
test_x = ss.fit_transform(test_x)

