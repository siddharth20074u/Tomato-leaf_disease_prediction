!pip install tensorflow-gpu
!nvidia-smi

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
from glob import glob

IMAGE_SIZE = [224, 224]
train_path = '/content/drive/My Drive/Tomato-leaf/train'
test_path = '/content/drive/My Drive/Tomato-leaf/valid'

inception = InceptionV3(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)

for layer in inception.layers:
  layer.trainable = False

folders = glob('/content/drive/My Drive/Tomato-leaf/train/*')

x = Flatten()(inception.output)
prediction = Dense(len(folders), activation = 'softmax')(x)

model = Model(inputs = inception.input, outputs = prediction)
model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/content/drive/My Drive/Tomato-leaf/train', target_size = (224, 224), batch_size = 32, class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('/content/drive/My Drive/Tomato-leaf/valid', target_size = (224, 224), batch_size = 32, class_mode = 'categorical')

r = model.fit_generator(training_set, validation_data = test_set, epochs = 10, steps_per_epoch = len(training_set), validation_steps = len(test_set))

import matplotlib.pyplot as plt

plt.plot(r.history['loss'], label = 'train loss')
plt.plot(r.history['val_loss'], label = 'val loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label = 'train acc')
plt.plot(r.history['val_accuracy'], label = 'val acc')
plt.legend()
plt.show()

y_pred = model.predict(test_set)
print(y_pred)

model.save(InceptionV3.h5)
