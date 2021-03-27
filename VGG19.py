 
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

 
# import the libraries as shown below

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
#import matplotlib.pyplot as plt

 
# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'Dataset/DOC/train'
valid_path = 'Dataset/DOC/test'


# %% [markdown]
#  Import the VGG16 library as shown below and add preprocessing layer to the front of VGG
#  Here we will be using imagenet weights

 

VGG19 = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


 
# don't train existing weights
for layer in VGG19.layers:
    layer.trainable = False

 
  # useful for getting number of output classes
folders = glob('Dataset/DOC/train/*')
folders

 
# our layers - you can add more if you want
x = Flatten()(VGG19.output)

 
len(folders)

 
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=VGG19.input, outputs=prediction)

 

# view the structure of the model
model.summary()


 
# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('Dataset/DOC/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

 
test_set = test_datagen.flow_from_directory('Dataset/DOC/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

 
# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=20,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

 
import matplotlib.pyplot as plt

 
# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

 
# save it as a h5 file


from tensorflow.keras.models import load_model

model.save('model_vgg19.h5')

 

y_pred = model.predict(test_set)
y_pred

 
import numpy as np
y_pred = np.argmax(y_pred, axis=1)

 
y_pred


 

# %%
def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor





# %%

test_image = load_image(r'D:\Pramod\AI\CNN\Transfer-Learning-master\Transfer-Learning-master\Dataset\DOC\validation\2.jpg')
prediction = xmodel.predict(test_image)
prediction

# %%

indexes = np.argmax(prediction, axis=1)
print(indexes)
training_set.class_indices

# %%
import os
import csv

for filename in os.listdir(r'D:\Pramod\AI\CNN\Transfer-Learning-master\Transfer-Learning-master\Dataset\DOC\validation'):
  test_image = load_image(filename)
  prediction = xmodel.predict(test_image)
  prediction
  indexes = np.argmax(prediction, axis=1)
  print(indexes)
  training_set.class_indices

# %%
