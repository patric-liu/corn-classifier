import numpy as np
import matplotlib.pyplot as plt
import keras
import os
from keras.preprocessing import image
from keras.applications import ResNet50
from keras.layers import Input
from keras.models import model_from_json
from keras.models import Model
from keras.utils import plot_model
from keras import models
from keras import layers


'''
Transfer Learning Tester

Loads model previously trained with transfer learning and runs custom images 
through the model to aid in model evaluation and debugging

'''

# build model
res_conv = ResNet50(weights='imagenet',
                    include_top=False,
                    input_shape=(224, 224, 3))

# FC layer/model reconstruction from JSON
with open('model_architecture.json', 'r') as f:
    fc = model_from_json(f.read())
# load weights into the FC model
fc.load_weights('model_weights.h5')

input_image = Input(shape=(224,224,3))

features = res_conv(input_image)
predictions = fc(features)

full_model = Model(inputs = input_image, outputs = predictions)


print('\n', '\n', '\n')

img_path = os.path.dirname(__file__) + '/manual-evaluation-dataset/example.png'

img = image.load_img(path = img_path, grayscale=False,
                     target_size=(224, 224, 3))

img = image.img_to_array(img) # coverts from PIL format to numpy array
img = img/255 
img_4dim = img[np.newaxis, :, :, :]

img_class = full_model.predict(img_4dim)

classname = img_class[0]


print("class:",classname)
plt.imshow(img)
#plt.title("predicted class:", classname)
plt.show()
