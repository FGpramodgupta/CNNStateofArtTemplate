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

IMAGE_SIZE = [224, 224]


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# %% [code]
# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('Dataset/DOC/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


classes=training_set.class_indices

from tensorflow.keras.models import load_model
xmodel=load_model('model_vgg19.h5')


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


test_image = load_image(r'D:\Pramod\AI\CNN\Transfer-Learning-master\Transfer-Learning-master\Dataset\DOC\validation\2.jpg')
prediction = xmodel.predict(test_image)
prediction


indexes = np.argmax(prediction, axis=1)
print(indexes)



key_list = list(classes.keys())
val_list = list(classes.values())




print()

import os
validationpath=r'D:\Pramod\AI\CNN\Transfer-Learning-master\Transfer-Learning-master\Dataset\DOC\validation'
predictionpath=r'D:\Pramod\AI\CNN\Transfer-Learning-master\Transfer-Learning-master\Dataset\DOC\prediction'
for filename in os.listdir(validationpath):
    test_image = load_image(validationpath +'\\' + filename)
    prediction = xmodel.predict(test_image)
    prediction
    indexes = np.argmax(prediction, axis=1)
    print(indexes)
    position = val_list.index(indexes)
    print(key_list[position])
    
    img = image.load_img(validationpath +'\\' + filename)
    img.save(predictionpath + '\\' + key_list[position] +'_vgg19_' + filename)
  
  
  

