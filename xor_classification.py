# -*- coding: utf-8 -*-
"""
KHAZEM
Khaled

"""

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

class_names = ['0 - FALSE', '1 - TRUE']

# Establishing the Dataset
features = np.array([[]], dtype='int32')
features.shape = (0, 2)
labels = np.array([], dtype='int32')
for i in range(1000):
  [a, b] = [np.random.randint(2), np.random.randint(2)]
  out = 0 if a == b else 1
  features = np.append(features,  [[a, b]], axis=0)
  labels = np.append(labels, out)

# Defining the layers
l1 = tf.keras.layers.Dense(units=2, 
                           input_shape=(2,),
                           activation='sigmoid')

l2 = tf.keras.layers.Dense(units=2,
                          activation='softmax')

# Building the neural net's model
model = tf.keras.Sequential([l1, l2])

# Setting up the parametres
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(0.1),
              metrics=['accuracy'])

# Training ...
history = model.fit(features, labels, epochs=20, verbose=False)
print("Finished training the model")

# Ploting the loss variation 
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

# Hard coding the validity of the net 
prediction_1 = np.around(model.predict(np.array([[0,0]])))
prediction_2 = np.around(model.predict(np.array([[1,1]])))
prediction_3 = np.around(model.predict(np.array([[0,1]])))
prediction_4 = np.around(model.predict(np.array([[1,0]])))

print(class_names[np.where(prediction_1[0] == 1)[0][0]])
print(class_names[np.where(prediction_2[0] == 1)[0][0]])
print(class_names[np.where(prediction_3[0] == 1)[0][0]])
print(class_names[np.where(prediction_4[0] == 1)[0][0]])
