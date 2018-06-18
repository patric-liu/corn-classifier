import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import to_categorical
import os
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications import ResNet50
from time import time
from keras.callbacks import TensorBoard


res_conv = ResNet50(weights='imagenet',
                 include_top=False,
                 input_shape=(224, 224, 3))
res_conv.summary()  # Display Resnet50 model structure

print('\n', '\n', '\n')

##########################################################################
debug = False
num_labels = 2  # number of categories for the data
epochs = 10000
learning_rate = 3e-6
fc_size = 128

nTrain = 250  # number of images to train on each epoch
nVal = 250  # numer of images to evaluate model on
batch_size = 25  # minibatch size
training_batch_size = 10
##########################################################################

if debug:
    nTrain, nVal, batch_size, num_labels = 15, 6, 3, 3
    path = os.path.dirname(__file__) + '/debug-dataset'
else:
    # Location of training data and validation data
    path = os.path.dirname(__file__) + '/clean-dataset-backup'

train_dir = path + '/train'
validation_dir = path + '/validation'


datagen = ImageDataGenerator(rescale = 1./255.) 

nTrain = nTrain - nTrain % batch_size
nVal = nVal - nVal % batch_size

if debug:
    nTrain, nVal, batch_size = 15, 6, 3


''' since this is using transfer learning, the downloaded model does not contain
the final fully connected later nor the output layer. Because the convolution 
weights remain untouched and will have the same output for the same images before 
and after training, we find the 'output' of these convolution layers for 
each of the training and validation images and save these to reuse during training
'''

# Find outputs for training data
train_features = np.zeros(shape=(nTrain, 1, 1, 2048)) 
train_labels = np.zeros(shape=(nTrain, num_labels)) 
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)
i = 0
for inputs_batch, labels_batch in train_generator:
    if i * batch_size >= nTrain:
        break
    print('[train] gathering features for batch',i)
    features_batch = res_conv.predict(inputs_batch)
    train_features[i * batch_size: (i + 1) * batch_size] = features_batch
    train_labels[i * batch_size: (i + 1) * batch_size] = labels_batch
    i += 1
train_features = np.reshape(train_features, (nTrain, 1 * 1 * 2048))

# find outputs for validation data
validation_features = np.zeros(shape=(nVal, 1, 1, 2048))
validation_labels = np.zeros(shape=(nVal, num_labels))
validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)
i = 0
for inputs_batch, labels_batch in validation_generator:
    if i * batch_size >= nVal:
        break
    print('[eval] gathering features for batch', i)
    features_batch = res_conv.predict(inputs_batch)
    validation_features[i * batch_size: (i + 1) * batch_size] = features_batch
    validation_labels[i * batch_size: (i + 1) * batch_size] = labels_batch
    i += 1

validation_features = np.reshape(validation_features, (nVal, 1 * 1 * 2048))

''' Now we add in a fully connected layer and output layer to complete the model
If we take the output of the convolution layers of the ResNet as the input to these
new layers, we are just creating a simple 1 hidden layer network
'''
from keras import models
from keras import layers
from keras import optimizers

# Create the model
model = models.Sequential()
model.add(layers.Dense(fc_size, activation='relu', input_dim=1 * 1 * 2048))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(num_labels, activation='softmax'))

# Choose the optimizer
model.compile(optimizer=optimizers.RMSprop(lr=learning_rate),
              loss='categorical_crossentropy',
              metrics=['acc'])

# terminal: tensorboard --logdir=logs/
# browser: http://localhost:6006/
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))


# Train the model
history = model.fit(train_features,
                    train_labels,
                    epochs=epochs,
                    batch_size=training_batch_size,
                    validation_data=(validation_features, validation_labels),
                    callbacks=[tensorboard])

# save model weights then architecture
from keras.models import model_from_json
print('saving')
model.save_weights('model_weights.h5')
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())

fnames = validation_generator.filenames
ground_truth = validation_generator.classes
label2index = validation_generator.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v, k) for k, v in label2index.items())
print(idx2label)


# Evaluate model on evaluation dataset
predictions = model.predict_classes(validation_features)
prob = model.predict(validation_features)
errors = np.where(predictions != ground_truth)[0]

# Number of errors BROKEN
# print("No of errors = {}/{}".format(len(errors), nVal))


# by extension also broken
'''for i in range(len(errors)):
    pred_class = np.argmax(prob[errors[i]])
    pred_label = idx2label[pred_class]

    print('Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        prob[errors[i]][pred_class]))

    original = load_img('{}/{}'.format(validation_dir, fnames[errors[i]]))
    plt.imshow(original)
    plt.show()
'''
