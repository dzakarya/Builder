import cv2
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import glob
import sys
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential, load_model
from keras import optimizers,Model
from keras.layers.advanced_activations import ELU,LeakyReLU
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Input, AveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from keras import backend as K
import csv
sys.path.append('..')
cd = os.getcwd()


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
            s = cv2.resize(imgfile, (224, 224))
            im = np.array(s)
            im = im.astype('float32')
            im = preprocess_input(im)
            face.append(im)
            label.append(nm)
            print(nm)
    X.extend(face)
    Y.extend(label)
    return np.asarray(X), np.asarray(Y)

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

def builder():
    aug = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.5,
        height_shift_range=0.5,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    test = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.5,
        height_shift_range=0.5,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    aug.fit(trainx)
    test.fit(testx)

    K.clear_session()
    baseModel = ResNet50(weights=None, include_top=False,
                         input_tensor=Input(shape=(224, 224, 3)))
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(units=1000)(headModel)
    headModel = ELU(0.3)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)
    print(model.summary())
    sgd = optimizers.SGD(lr=0.01)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['accuracy'])
    Monitor = EarlyStopping(monitor='val_loss', patience=20, min_delta=0.001, verbose=1, restore_best_weights=True)
    hist = model.fit_generator(aug.flow(trainx, labtrny, batch_size=32), epochs=200, validation_data=test.flow(testx,labtesy,batch_size=32),
                      callbacks=[Monitor],verbose=2,steps_per_epoch=len(trainx)//32)
    newfit = min(hist.history['val_loss'])
    newacc = max(hist.history['val_acc'])
    print('New Model Validation Fitness: ', newfit)
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history['val_acc'])
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
    plt.show()
    model.save("res.h5")
    fit = hist.history['val_loss']
    accs = hist.history['val_accuracy']
    np.savetxt("loss_relu.txt", fit, delimiter=",")
    np.savetxt("acc_relu.txt", accs, delimiter=",")


if __name__ == '__main__':
    builder()