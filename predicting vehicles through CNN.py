import tensorflow as tf
from tensorflow.keras import datasets, layers, models import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from keras.layers.core import Dropout, Activation, Dense, Flatten from keras.models import Sequential

(x_train,y_train),(x_test,y_test) = datasets.cifar10.load_data() 

#below airplane=0, automobile=l, bird=2, etc are by deflaut
classes= ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

y_train = y_train.reshape(-1,)

#in order to let the pixels RGB color range from 0 to 255 let divide them by 255
x_train = x_train/255
x_test = x_test/255

y_test[:7]
#here frog index number equal to 6 
plt.imshow(x_test[7])

#Model creation
model= models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

#fitting to train data
model.fit(x_train, y_train, epochs=10) 

#testing data
model.fit(x_test,y_test)

#checking the results
y_predicted = model.predict(x_test)
y_pred = [np.argmax(element) for element in y_predicted]
y_pred[:5]
#output:
[3,	8,	8,   0,	6]


classes[3]
>>cat 

classes[8]
>>ship

classes[8]
>>ship

classes[0]
>>airplane

classes[6]
>>frog

#Checking the results with test values
y_test[:5]

#output:
array([[3],
[8],
[8],
[0],
[6]], dtype=uint8)

# above all are perfectly predicted (y_pred[:5] exactly matched with y_test[:5])
##Here it is predicted with perfect accuracy i.e is 80.68% Training, Testing 71.75%
