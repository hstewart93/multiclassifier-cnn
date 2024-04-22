import datetime

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from keras import applications
from keras.preprocessing.image import img_to_array, load_img
import time


# Default dimensions we found online
img_width, img_height = 224, 224

# Create a bottleneck file
top_model_weights_path = "bottleneck_fc_model.h5"
# loading up our datasets
train_data_dir = "data/train"
validation_data_dir = "data/validation"

# number of epochs to train top model
epochs = 7  # this has been changed after multiple model run
# batch size used by flow_from_directory and predict_generator
batch_size = 50


# # Loading vgc16 model
vgg16 = applications.VGG16(include_top=False, weights="imagenet")

datagen = ImageDataGenerator(rescale=1. / 255)
# needed to create the bottleneck .npy files

# training data
generator_top_train = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
)

nb_train_samples = len(generator_top_train.filenames)
num_classes = len(generator_top_train.class_indices)

# load the bottleneck features saved earlier
train_data = np.load("bottleneck_features_train.npy")

# get the class labels for the training data, in the original order
train_labels = generator_top_train.classes

# convert the training labels to categorical vectors
train_labels = to_categorical(train_labels, num_classes=num_classes)


# Validation data
generator_top_validation = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
)

nb_validation_samples = len(generator_top_train.filenames)

# load the bottleneck features saved earlier
validation_data = np.load("bottleneck_features_validation.npy")
# import ipdb; ipdb.set_trace(context=25)
validation_labels = generator_top_validation.classes
validation_labels = to_categorical(validation_labels, num_classes=num_classes)


# This is the best model we found. For additional models, check out I_notebook.ipynb
start = datetime.datetime.now()
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(100, activation=keras.layers.LeakyReLU(alpha=0.3)))
model.add(Dropout(0.5))
model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3)))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation="softmax"))
model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=["acc"],
)
history = model.fit(
    train_data,
    train_labels,
    epochs=7,
    batch_size=batch_size,
    validation_data=(validation_data, validation_labels),
)
model.save_weights(top_model_weights_path)
(eval_loss, eval_accuracy) = model.evaluate(
    validation_data, validation_labels, batch_size=batch_size, verbose=1
)
print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
print("[INFO] Loss: {}".format(eval_loss))
end = datetime.datetime.now()
elapsed = end-start
print("Time: ", elapsed)

# import ipdb; ipdb.set_trace(context=25)


# #Graphing our training and validation
# acc = history.history["acc"]
# val_acc = history.history["val_acc"]
# loss = history.history["loss"]
# val_loss = history.history["val_loss"]
# epochs = range(len(acc))
# plt.plot(epochs, acc, "r", label="Training acc")
# plt.plot(epochs, val_acc, "b", label="Validation acc")
# plt.title("Training and validation accuracy")
# plt.ylabel("accuracy")
# plt.xlabel("epoch")
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, "r", label="Training loss")
# plt.plot(epochs, val_loss, "b", label="Validation loss")
# plt.title("Training and validation loss")
# plt.ylabel("loss")
# plt.xlabel("epoch")
# plt.legend()
# plt.show()


def read_image(file_path):
    print("[INFO] loading and preprocessing image…")
    image = load_img(file_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.
    return image


def test_single_image(path):
    animals = ["cane", "cavallo", "elefante", "farfalla", "gallina", "gatto", "mucca", "pecora", "ragno", "scoiattolo"]
    images = read_image(path)
    time.sleep(.5)
    bt_prediction = vgg16.predict(images)
    preds = model.predict_proba(bt_prediction)
    for idx, animal, x in zip(range(0,6), animals , preds[0]):
        print("ID: {}, Label: {} {}%".format(idx, animal, round(x*100,2) ))
    print('Final Decision:')
    time.sleep(.5)
    for x in range(3):
        print('.'*(x+1))
        time.sleep(.2)
    class_predicted = model.predict_classes(bt_prediction)
    class_dictionary = generator_top_train.class_indices
    inv_map = {v: k for k, v in class_dictionary.items()}
    print("ID: {}, Label: {}".format(class_predicted[0], inv_map[class_predicted[0]]))
    return load_img(path)


path = "data/test/WhatsApp Image 2020-06-26 at 19.26.53.jpeg"
test_single_image(path)
