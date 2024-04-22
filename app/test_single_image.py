# import numpy as np
# from keras import applications
# from keras.preprocessing.image import img_to_array, load_img
# import time
#
# # Loading vgc16 model
# vgg16 = applications.VGG16(include_top=False, weights="imagenet")
#
# # This is the best model we found. For additional models, check out I_notebook.ipynb
# start = datetime.datetime.now()
# model = Sequential()
# model.add(Flatten(input_shape=train_data.shape[1:]))
# model.add(Dense(100, activation=keras.layers.LeakyReLU(alpha=0.3)))
# model.add(Dropout(0.5))
# model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3)))
# model.add(Dropout(0.3))
# model.add(Dense(num_classes, activation="softmax"))
# model.compile(
#     loss="categorical_crossentropy",
#     optimizer=optimizers.RMSprop(lr=1e-4),
#     metrics=["acc"],
# )
# history = model.fit(
#     train_data,
#     train_labels,
#     epochs=7,
#     batch_size=batch_size,
#     validation_data=(validation_data, validation_labels),
# )
# model.save_weights(top_model_weights_path)
# (eval_loss, eval_accuracy) = model.evaluate(
#     validation_data, validation_labels, batch_size=batch_size, verbose=1
# )
#
# def read_image(file_path):
#     print("[INFO] loading and preprocessing image…")
#     image = load_img(file_path, target_size=(224, 224))
#     image = img_to_array(image)
#     image = np.expand_dims(image, axis=0)
#     image /= 255.
#     return image
#
#
# def test_single_image(file_path):
#     animals = ["cane", "cavallo", "elefante", "farfalla", "gallina", "gatto", "mucca", "pecora", "ragno", "scoiattolo"]
#     images = read_image(file_path)
#     time.sleep(.5)
#     bt_prediction = vgg16.predict(images)
#     preds = model.predict_proba(bt_prediction)
#     for idx, animal, x in zip(range(0,6), animals , preds[0]):
#     print(“ID: {}, Label: {} {}%”.format(idx, animal, round(x*100,2) ))
#     print(‘Final Decision:’)
#     time.sleep(.5)
#     for x in range(3):
#     print(‘.’*(x+1))
#     time.sleep(.2)
#     class_predicted = model.predict_classes(bt_prediction)
#     class_dictionary = generator_top.class_indices
#     inv_map = {v: k for k, v in class_dictionary.items()}
#     print(“ID: {}, Label: {}”.format(class_predicted[0],  inv_map[class_predicted[0]]))
#     return load_img(path)
#
# path = ‘data/test/yourpicturename’
# test_single_image(path)
