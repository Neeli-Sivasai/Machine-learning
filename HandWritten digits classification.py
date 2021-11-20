from tensorflow import keras
from keras.models import Sequential, load_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import numpy as np
from sklearn.metrics import accuracy_score

(x_train,y_train), (x_test,y_test)= mnist.load_data()

x_train_changed = x_train.reshape(len(x_train) , 784)
print(x_train_changed. shape)
x_test_changed = x_test.reshape(len(x_test),784)
#print (x_test_changed. shape)

x_train = x_train/255
x_test = x_test/255

#model creation
model = Sequential()
model.add(Dense((512),input_shape = (784,)))
model.add(Activation('relu'))
model.add(Dense(512) )
model.add(Activation( 'relu'))
model.add(Dense(10) )
model.add(Activation('softmax' ))


model.compile(optimizer='adam', loss= 'Sparse categorical_crossentropy', metrics = ['accuracy'])
#fitting model
model.fit(x_train, y_train)
y_predict = model.predict(x_test_changed)
#checking accuracy
accuracy_score(y_test,y_predict)

