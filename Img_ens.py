# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 11:22:30 2022

@author: tammisetti
"""

#import Augmentor
#p= Augmentor.Pipeline("C:\\Users\\Tammisetti\\image ensemble\\kvasir-dataset", 
                      output_directory='C:\\Users\\Tammisetti\\image ensemble\\kvasir-dataset\\data')
#p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
#p.zoom(probability=0.3, min_factor=1.1, max_factor=1.6)
#p.sample(4000)


import pathlib
import tensorflow as tf 

data_dir = pathlib.Path('C:\\Users\\Tammisetti\\image ensemble\\kvasir-dataset\\data')
img_height =224
img_width =224

train_d =tf.keras.utils.image_dataset_from_directory(data_dir,validation_split=0.2,
                                                     subset='training',
                                                     seed=123,
                                                     image_size = (img_height,img_width),
                                                     batch_size=32)

val_d = tf.keras.utils.image_dataset_from_directory(data_dir,validation_split=0.2,
                                                     subset='validation',
                                                     seed=123,
                                                     image_size = (img_height,img_width),
                                                     batch_size=32)
normalization_layer =tf.keras.layers.Rescaling(1./255)

import numpy as np 
normalized_ds= train_d.map(lambda x, y: (normalization_layer(x),y))
image_batch, label_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image),np.max(first_image))

AUTOTUNE= tf.data.AUTOTUNE
train_d= train_d.cache().shuffle(100).prefetch(buffer_size= AUTOTUNE)
val_d = val_d.cache().prefetch(buffer_size=AUTOTUNE)


from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model 
from keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Dropout
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
base_model = InceptionV3(input_shape=(224,224,3),
                         weights='imagenet',
                         include_top= False)

#freez first 10 layers

for layer in base_model.layers[:10]:
    layer.trainable = False
x= base_model.output
x= GlobalAveragePooling2D()(x)
x= Dense(512, activation='relu')(x)
x= Dropout(0.4)(x)
predictions = Dense(8, activation='softmax')(x)
model1= Model(inputs=base_model.inputs,outputs=predictions)

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model 
from keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Dropout
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
base_model1 = VGG16(input_shape=(224,224,3),
                         weights='imagenet',
                         include_top= False)

for layer in base_model.layers[:10]:
    layer.trainable = False
x= base_model1.output
x= GlobalAveragePooling2D()(x)
x= Dense(512, activation='relu')(x)
x= Dropout(0.4)(x)
predictions = Dense(8, activation='softmax')(x)
model2= Model(inputs=base_model1.inputs,outputs=predictions)


from tensorflow.keras.callbacks import ModelCheckpoint

model_filepath= 'saved_models/model1.hdf5'
checkpoint = ModelCheckpoint(filepath= model_filepath,
                             monitor='val_accuracy',
                             mode='max',
                             save_best_only=True,
                             verbose=1)

model1.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])
model2.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])



history1= model1.fit(train_d,
                     validation_data=val_d,
                     epochs=2, callbacks=[checkpoint])

#model1.save('saved_models/model1.hdf5')


import matplotlib.pyplot as plt
acc= history1.history['accuracy']
val_acc= history1.history['val_accuracy']
loss=history1.history['loss']
val_loss=history1.history['val_loss']

epochs=range(len(acc))

plt.plot(epochs,acc,'r', label= 'Training accurcy')
plt.plot(epochs,val_acc,'g', label= 'validation accurcy')
plt.title('Training and validation acc')
plt.legend()
plt.figure()




model_filepath= 'saved_models/model2.hdf5'
checkpoint = ModelCheckpoint(filepath= model_filepath,
                             monitor='val_accuracy',
                             mode='max',
                             save_best_only=True,
                             verbose=1)




history2= model2.fit(train_d,
                     validation_data=val_d,
                     epochs=2, callbacks=[checkpoint])




import matplotlib.pyplot as plt
acc= history2.history['accuracy']
val_acc= history2.history['val_accuracy']
loss=history2.history['loss']
val_loss=history2.history['val_loss']

epochs=range(len(acc))

plt.plot(epochs,acc,'r', label= 'Training accurcy')
plt.plot(epochs,val_acc,'g', label= 'validation accurcy')
plt.title('Training and validation acc')
plt.legend()
plt.figure()


from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Average
model_1 = load_model("C:\\Users\\Tammisetti\\image ensemble\\kvasir-dataset\\saved_models\\model1.hdf5")
model_1 = Model(inputs=model_1.inputs,
                outputs=model_1.outputs,
                name='model_1')
model_2 = load_model("C:\\Users\\Tammisetti\\image ensemble\\kvasir-dataset\\saved_models\\model2.hdf5")
model_2 = Model(inputs=model_2.inputs,
                outputs=model_2.outputs,
                name='model_2')
models = [model_1, model_2]
model_input = Input(shape=(224, 224, 3))
model_outputs = [model(model_input) for model in models]
ensemble_output = Average()(model_outputs)
ensemble_model = Model(inputs=model_input, outputs=ensemble_output, name='ensemble')

ensemble_model.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])



'''
model_filepath= 'saved_models/modelANS.hdf5'
checkpoint = ModelCheckpoint(filepath= model_filepath,
                             monitor='val_accuracy',
                             mode='max',
                             save_best_only=True,
                             verbose=1)
'''

history3= ensemble_model.fit(train_d,
                     validation_data=val_d,
                     epochs=2)

acc= history3.history['accuracy']
val_acc= history3.history['val_accuracy']
loss=history3.history['loss']
val_loss=history3.history['val_loss']

epochs=range(len(acc))

plt.plot(epochs,acc,'r', label= 'Training accurcy')
plt.plot(epochs,val_acc,'g', label= 'validation accurcy')
plt.title('Training and validation acc')
plt.legend()
plt.figure()