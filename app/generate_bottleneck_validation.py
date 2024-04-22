"""Generate numpy bottleneck file for training set"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
import math
import datetime

# Default dimensions we found online
img_width, img_height = 224, 224

# Create a bottleneck file
top_model_weights_path = "bottleneck_fc_model.h5"
# loading up our datasetsv
validation_data_dir = "data/validation"

# number of epochs to train top model
epochs = 7  # this has been changed after multiple model run
# batch size used by flow_from_directory and predict_generator
batch_size = 50

# Loading vgc16 model
vgg16 = applications.VGG16(include_top=False, weights="imagenet")
datagen = ImageDataGenerator(rescale=1. / 255)
# needed to create the bottleneck .npy files

#################################################################
# __this can take an hour and half to run so only run it once.
# once the npy files have been created, no need to run again. Convert this cell to a code cell to run.__
start = datetime.datetime.now()

generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

nb_validation_samples = len(generator.filenames)
num_classes = len(generator.class_indices)

predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))

bottleneck_features_validation = vgg16.predict_generator(generator, predict_size_validation)

np.save("bottleneck_features_validation.npy", bottleneck_features_validation)
end = datetime.datetime.now()
elapsed = end - start
print("Time: ", elapsed)
