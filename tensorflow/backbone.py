# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ### Image classification; Backbone of CNN

# * learning purpose
#     * Understanding the concept of Transfer Learning by knowing the types and concepts of Backbone model, which is a pre-learning model.
#     * It'll bring in basic Backbone models like VGG and ResNet and use them.
#     * It newly learns as much as the layer wanting the Backbone model and use.
#     * As much as I want image classify with Backbone model by Transfering learning.

# +
# Required Message Alert Message Output x
import warnings
warnings.filterwarnings("ignore")

print("set")
# -

# version of tensorflow for model design check
import tensorflow as tf
print(tf.__version__)

# **1, prepare the dataset for learning model**

# +
# load datasets of tensorflow_datasets
import tensorflow_datasets as tfds

tfds.__version__
# -

# [Datasets provided by TensorFlow](https://www.tensorflow.org/datasets/catalog/overview)

# load cats_vs_dogs datasets of tensorflow_datasets
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True)

# each dataset check
print(raw_train)
print(raw_validation)
print(raw_test)

# : All datasets are (image, label) shape. = ((None, None, 3), ()) --> (None, None, 3) == (height, width, channel) three-dimension Images, () == shape of label.

# **2. Data visualization and Dtaa pre-processing**

# +
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

print('set')
# -

# **take function provided tf.data.Dataset**

# : This function extracts as many data as received as the argument, generates a new dataset instance, and returns it.

# +
plt.figure(figsize=(10, 5))

get_label_name = metadata.features['label'].int2str

for idx, (image, label) in enumerate(raw_train.take(10)):  # take 10 data
    plt.subplot(2, 5, idx+1)
    plt.imshow(image)
    plt.title(f'label {label}: {get_label_name(label)}')
    plt.axis('off')
# -

# **Image resize**

# format_example(): format unification.

# * Type Casting
#     * Type Casting, called type-transformation, means changing the type to another data type. Using a float() to convert an integer type into a float type is an example of Type Casting.

# +
IMG_SIZE = 160 # resizing Image size

def format_example(image, label):
    image = tf.cast(image, tf.float32) # image = float(image) tensorflow version of same type casting
    image = (image/127.5) - 1   # revise scale of same pixel values
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

print('set')
# -

# : Values between 0~255 are divided into the intermediate value 127.5 and 1 is taken out. Therefore, the float value between -1 ~ 1 is.

# +
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

print(train)
print(validation)
print(test)

# +
plt.figure(figsize=(10, 5))


get_label_name = metadata.features['label'].int2str

for idx, (image, label) in enumerate(train.take(10)):
    plt.subplot(2, 5, idx + 1)
    image = (image + 1) / 2
    plt.imshow(image)
    plt.title(f'label {label}: {get_label_name(label)}')
    plt.axis('off')
# -

# : When visualizing the image with matplotlib, all the pixel values should be positve, so the pixel values -1 to 1 are added 1 and divied into 2 and converted into values 0 to 1. 

# **3. Model Structure Design using Tensorflow**

# +
# Load the functions required for model generation.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

print('set')
# -

model = Sequential([
    Conv2D(filters=16, kernel_size=3,padding='same', activation='relu', input_shape=(160, 160, 3)),
    MaxPooling2D(),
    Conv2D(filters=32, kernel_size=3, padding='same'),
    MaxPooling2D(),
    Conv2D(filters=64, kernel_size=3, padding='same'),
    MaxPooling2D(),
    Flatten(),
    Dense(units=512, activation='relu'),
    Dense(units=2, activation='softmax')
])

model.summary()

# : The first dimension represents the number of data. It's marked by the None symbol, "Undetermined Number."None indicates that different numbers of inputs can be entered into the model, depending on the batch size.
# One size of data is three dimensions (height, width, channel). As we pass through the six layers, the height and width become smaller and the channel grows larger, and then the shape decreases to one number: 25,600 (20x20x64) by meeting the flatten layer.
# The network, which has a smaller and smaller feature map output from CNN (Convolutional Neural Net), and then a one-dimensional shape through Flatten and Dense layers, is the most representative form of the deep learning model using CNN.

# ![%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202022-01-15%20144140.jpg](attachment:%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202022-01-15%20144140.jpg)

# [출처: Gradient-Based Learning Applied to Docoument Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

# : When an image is enterd like the left side, the image becomes longer and longer through the Convolutional operation, and when the Flatten layer meets, it is spread out in one line like the right side. It;s a three-dimensional image that's spread out in one dimension.

# **★★Understaning Flatten**

# +
import numpy as np

image = np.array([[1, 2], [3, 4]])
print(image.shape)
image
# -

image.flatten()

# : All numbers are in row, and passing through the flatten layer in the model is like spreading all the numbers in a row. <br/>
# <br/>
# Then, the Dense layer is reduced to 512 bides, and the final output generates a probaility distribution of only new numbers. These two numbers is a puppy and the probability that it is a cat.<br/>
# <br/>
# To summarize, the deep learning model is a function that inputs a three-dimensional image of (160, 160, 3)size, changes the shape through several layers, and finally outputs a few numbers.

# **4. After compiling model, learn it**

# +
'''
The paramether called learning rate is set to 0.0001,
and the model is compiled to transform into a form that can be learned.
'''
learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

print('set')
# -

# **To do the compile, three things are needed: optimizer, loss, and metrics.**

# * Optimizer determines how to make learning; it also calls it an optimization function because it determines how to optimize it
# * Loss determines the direction the model must learn.In this problem, the output of the model is set as a probability distribution for whether the input image is a cat or a puppy, so that if the input image is a cat, the output of the model is close to [1.0, 0.0], and if the image is a puppy, it is closer to [0.0, 1.0].
# * Metrics is a measure of the performance of the model.When solving classification problems, indicators that can evaluate performance include accuracy, precision, and recall; here, using accuracy

# +
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

print('set')
# -

# : It will make train_batches, validation_batches, and test_batches that randomly sprinkle 32 data according to BATCH_SIZE.Train_batches will continue to provide 32 randomly drawn from the entire data so that the model can be constantly learned.

# +
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

print('set')

# +
# Take only one batch out of the train_batches to check the data
for image_batch, label_batch in train_batches.take(1):
    pass

image_batch.shape, label_batch.shape
# -

# **Performance validation**

# +
validation_steps = 20
loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:2f}".format(accuracy0))
# -

EPOCHS = 10
history = model.fit(train_batches,
                   epochs=EPOCHS,
                   validation_data=validation_batches)

# * The first accuracy of the training dataset is.
#     * Accuracy of data being learned.
# * The second val_accuracy is the accuracy of the validation dataset.
#     * Accuracy of data that is not being learned, that is, not seen in the corresponding learning step.

# +
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
# -

# : Training accuracy rises steadily, but validation accuracy does not seem to exceed any limit. Even in the loss graph, training loss is steadily reduced, but the valuation loss value is increased again after a certain moment.

# Overfitting.To improve the performance of the model properly, the performance should be good for "unlearning" data, but as we continue to study with only training data, it becomes excessively fit for the data and the generalization ability is lowered.

# +
# checking a model prediction value
for image_batch, label_batch in test_batches.take(1):
    images = image_batch
    labels = label_batch
    predictions = model.predict(image_batch)
    pass

predictions
# -

predictions = np.argmax(predictions, axis=1)
predictions

# +
plt.figure(figsize=(20, 12))

for idx, (image, label, prediction) in enumerate(zip(images, labels, predictions)):
    plt.subplot(4, 8, idx+1)
    image = (image + 1) / 2
    plt.imshow(image)
    correct = label == prediction
    title = f'real: {label} / pred :{prediction}\n {correct}!'
    if not correct:
        plt.title(title, fontdict={'color': 'red'})
    else:
        plt.title(title, fontdict={'color': 'blue'})
    plt.axis('off')

# +
count = 0
for image, label, prediction in zip(images, labels, predictions):
    correct = label == prediction
    if correct:
        count = count + 1
        
print(count / 32 * 100)
# -
# **5. Transfer learning - VGG16 Model**


# +
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model VGG16
base_model = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
                                         include_top=False,
                                         weights='imagenet')
# -

image_batch.shape

# feature vector
feature_batch = base_model(image_batch)
feature_batch.shape

base_model.summary()

# ![%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202022-01-17%20205618.jpg](attachment:%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202022-01-17%20205618.jpg)

# [출처:  VGG16 – Convolutional Network for Classification and Detection](https://neurohive.io/en/popular-networks/vgg16/)

# : The VGG16 model has the same structure as above.

# However, because this model will be transfer learning without using it, Dense Layers will not be used and will be made.

# ![%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202022-01-17%20210058.jpg](attachment:%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202022-01-17%20210058.jpg)

# <https://www.researchgate.net/figure/Types-of-pooling-d-Fully-Connected-Layer-At-the-end-of-a-convolutional-neural-network_fig3_337105858>

feature_batch.shape

# : The vectors that can be input into the solution- Fully connected layer must be one-dimentional, so this three-dimentional vector must be converted into one-dimensional vectors.

# **!!! Global Average Pooling**

# ![%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202022-01-17%20210948.jpg](attachment:%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202022-01-17%20210948.jpg)

# [출처: Global Average Pooling](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/blocks/global-average-pooling-2d)

# : Global Average Pooling is a technique that reduces the average of two-dimensional arrays stacked in layer to one value when there is a three-dimensional vector as above.
# <br/>
# <br/>
# In other words, the method of reducing the dimension of the vector by using the mean value is called Global Average Pooling.
#

# +
# Global Average Pooling
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

# +
dense_layer = tf.keras.layers.Dense(512, activation='relu')
prediction_layer = tf.keras.layers.Dense(2, activation='softmax')

# feature_batch_averag가 dense_layer를 거친 결과가 다시 prediction_layer를 거치게 되면
prediction_batch = prediction_layer(dense_layer(feature_batch_average))  
print(prediction_batch.shape)
# -

# * [the reason why using the activation function](https://ganghee-lee.tistory.com/30)
# * [activation function of deep learning](https://ynebula.tistory.com/42)

# : After the feature is extracted by being inputted into VGG16 and base_model, which will extract the feature from the image at first, the feature vector passes through the global_average_layer and finally passes to the prediction_layer, predicting whether it is a puppy or a cat

# +
base_model.trainable = False  # the base_model will not do the learning, so the trainable variable
                              # that determines whether to learn is specified as False.
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    dense_layer,
    prediction_layer
])    

model.summary()
# -

# **6. Final Learning**

# +
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
validation_steps=20
loss0, accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

# +
EPOCHS = 5  # It converges much faster than befor, so 5Epoch

history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)

# +
# visualizating
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
# -

# **Prediction result confirmation**

# +
for image_batch, label_batch in test_batches.take(1):
    images = image_batch
    labels = label_batch
    predictions = model.predict(image_batch)
    pass

predictions
# -

import numpy as np
predictions = np.argmax(predictions, axis=1)
predictions

# +
plt.figure(figsize=(20, 12))

for idx, (image, label, prediction) in enumerate(zip(images, labels, predictions)):
    plt.subplot(4, 8, idx+1)
    image = (image + 1) / 2
    plt.imshow(image)
    correct = label == prediction
    title = f'real: {label} / pred :{prediction}\n {correct}!'
    if not correct:
        plt.title(title, fontdict={'color': 'red'})
    else:
        plt.title(title, fontdict={'color': 'blue'})
    plt.axis('off')
# -

# : Surely using a well-educated model worked well.

# +
count = 0
for image, label, prediction in zip (images, labels, predictions):
    correct = label == prediction
    if correct:
        count = count + 1
        
print(count / 32 * 100)
