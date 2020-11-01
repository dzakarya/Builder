from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers,optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import os
from sklearn.model_selection import train_test_split
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
cd = os.getcwd()

size = 300

def load_img(dir1):
    face = []
    label = []
    X = []
    Y = []
    for subdir in os.listdir(dir1):
        path = dir1 + subdir
        for img in os.listdir(path + '/'):
            nm = os.path.splitext(os.path.basename(subdir))[0]
            imgfile = cv2.imread(path + '/' + img)
            s = cv2.resize(imgfile, (size, size))
            im = np.array(s)
            im = im.astype('float32')
            face.append(im)
            label.append(nm)
            print(nm)
    X.extend(face)
    Y.extend(label)
    return np.asarray(X),np.asarray(Y)

def encode(y):
    lab = []
    for label in y:
        if label == "safe":
            x = [1, 0]
        else:
            x = [0, 1]
        lab.append(x)
    return np.asarray(lab)

x,y = load_img("worker/")
trainx, testx, trainy, testy = train_test_split(x,y,test_size=0.2,random_state=0)
labtrny = encode(trainy)
labtesy = encode(testy)

img_augmentation = Sequential(
    [
        preprocessing.RandomRotation(factor=0.15),
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

lr_schedule = ExponentialDecay(initial_learning_rate=0.005,
                               decay_steps=10000,
                               decay_rate=0.9)

inputs = layers.Input(shape=(size, size, 3))
x = img_augmentation(inputs)
baseModel = EfficientNetB3(include_top=True, classes=2, weights=None)(inputs)
model = tf.keras.Model(inputs,baseModel)
print(model.summary())
sgd = optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['accuracy'])
Monitor = EarlyStopping(monitor='val_loss', patience=30, min_delta=0.01, verbose=1, restore_best_weights=True)
hist = model.fit(x=trainx,y=labtrny,batch_size=16,epochs=200,validation_data=(testx,labtesy),callbacks=[Monitor],
                 verbose=2)

model.save("enetapd.h5")
newfit = min(hist.history['val_loss'])
newacc = max(hist.history['val_accuracy'])
print('New Model Validation Fitness: ', newfit)
plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
plt.show()