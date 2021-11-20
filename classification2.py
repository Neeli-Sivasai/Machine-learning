import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from keras.layers.core import Dropout, Activation, Dense, Flatten
from keras.models import Sequential
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test,y_test) = datasets.fashion_mnist.load_data()

#to see shape of the data use fallowing code
#print(x_train. shape)
#print(y_train. shape)

#to get RGB pixel value in 0 to 1 let divide them by 255
x_train = x_train/255
x_test = x_test/255

classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Boot', 'Ankle boot']

model = Sequential()
model.add(Flatten(input_shape = (28,28)))
model.add(Dense(150))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss = "sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=10)

y_predict = model.predict(x_test)
y_predicted = [np.argmax(element) for element in y_predict]

plt.matshow(x_test[0])

accuracy_score(y_test,Y_predict, noramalize = False)
 



