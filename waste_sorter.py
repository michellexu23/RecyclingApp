BATCH_SIZE = 32
LEARNING_RATE = .001
EPOCHS = 64

VALIDATION_ACCURACY = []
VALIDATION_LOSS = []
TEST_ACCURACY = []
TEST_LOSS = []

import numpy as np
import os
import random

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.model_selection import StratifiedKFold

from keras.models import Model
from keras import layers
from keras import optimizers
from keras import losses

from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

#split indices for n samples into train and test indices with random sampling
def split_indices(n, train_split):    
    all = list(range(1,n+1))
    random.seed(2020)
    train = random.sample(list(range(1,n+1)),int(train_split*n))
    test = list(set(all)-set(train))
    return train, test

#load jpg image file and convert to numpy array
def load_image(filepath):
    img = image.load_img(filepath)
    return image.img_to_array(img) 

#prepare train and test data as numpy arrays, y output is a class vector
def build_dataset(data_dir, classes, train_split):
    X_train, Y_train, X_test, Y_test = [], [], [], []

    for i, c in enumerate(classes):
        class_path = os.path.join(data_dir, c)
        
        n = len(os.listdir(class_path))
        train_ind, test_ind = split_indices(n, train_split)
        
        ntrain = int(train_split*n)
        y_train = [i] * ntrain
        y_test = [i] * (n-ntrain)
        Y_train = np.append(Y_train, y_train)
        Y_test = np.append(Y_test, y_test)
     
        for j in train_ind:
            img = load_image(os.path.join(class_path, c + str(j) + ".jpg"))
            X_train.append(img)
        for j in test_ind:
            img = load_image(os.path.join(class_path, c + str(j) + ".jpg"))
            X_test.append(img)
    X_train = np.array(X_train)
    X_test = np.array(X_test)    
    #print(X_train[:10], Y_train[:10], X_test[:10], Y_test[:10])

    return X_train, Y_train, X_test, Y_test

def build_model():
    # load without classifier layers
    model = ResNet50(include_top=False, input_shape=(384, 512, 3))
    # add new classifier layers
    flat1 = layers.Flatten()(model.layers[-1].output)
    class1 = layers.Dense(1024, activation='relu')(flat1)
    output = layers.Dense(6, activation='softmax')(class1)

    model = Model(inputs=model.inputs, outputs=output)
    return model

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(EPOCHS, acc, label='Training Accuracy')
    plt.plot(EPOCHS, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.figure()
    plt.plot(EPOCHS, loss, label='Training Loss')
    plt.plot(EPOCHS, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

save_dir = './saved_models/'

classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
#80-20 training-test split
X_train, Y_train, X_test, Y_test = build_dataset('dataset-resized', classes, .8)

idg_aug = ImageDataGenerator(rotation_range=40,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               zoom_range=0.3,
                               fill_mode='nearest',
                               horizontal_flip=True,
                               preprocessing_function=preprocess_input)
idg = ImageDataGenerator(preprocessing_function=preprocess_input)

#5-fold stratified cross validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
fold_num = 1
for train_ind, val_ind in kf.split(np.zeros(len(X_train)), Y_train):
    x_train = X_train[train_ind]
    y_train = keras.utils.to_categorical(Y_train[train_ind], len(classes))
    x_val = X_train[val_ind]
    y_val = keras.utils.to_categorical(Y_train[val_ind], len(classes))

    ntrain = len(x_train)
    nval = len(x_val) 

    train_gen = idg_aug.flow(x_train, y_train, BATCH_SIZE)
    val_gen = idg.flow(x_val, y_val, BATCH_SIZE)

    model = build_model()
    model.compile(loss=losses.CategoricalCrossentropy(from_logits=True),
                  optimizer=optimizers.Adam(LEARNING_RATE),
                  metrics=['accuracy'])

    #save best model
    output_dir = save_dir+str(fold_num)+'.h5'
    checkpoint = ModelCheckpoint(output_dir,
                           monitor='val_accuracy',
                           save_best_only=True, mode='max')
    callbacks = [checkpoint]

    history = model.fit(train_gen,
                        steps_per_epoch=ntrain // BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=callbacks,
                        validation_data = val_gen,
                        validation_steps=nval // BATCH_SIZE)
    
    plot_history(history)
    
    #load best model
    model.load_weights(output_dir)
    results = model.evaluate(val_gen)
    results = dict(zip(model.metrics_names, results))
    VALIDATION_ACCURACY.append(results['accuracy'])
    VALIDATION_LOSS.append(results['loss'])
     
    test_gen = idg.flow(X_test, keras.utils.to_categorical(Y_test))
    test_results = model.evaluate(test_gen)
    test_results = dict(zip(model.metrics_names, test_results))
    TEST_ACCURACY.append(results['accuracy'])
    TEST_LOSS.append(results['loss'])
    
    keras.backend.clear_session()

    fold_num+=1
