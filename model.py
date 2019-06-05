import keras
import tensorflow as tf


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


model = Sequential()
model.add(Conv2D(filters = (32),kernel_size=(4,4),input_shape=(64,64,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(4,4),activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128,(4,4),activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(256,(4,4),activation="relu"))
model.add(MaxPooling2D(2,2))


model.add(Flatten())
model.add(Dense(32,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])


from keras.preprocessing.image import ImageDataGenerator
from IPython.display import display
from PIL import Image




train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.3,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data_set =train_datagen.flow_from_directory("/Users/user/Desktop/Samsung:Nokia/trainset",
                                                  target_size=(64,64),
                                                  batch_size=50,
                                                  class_mode="binary")
print(train_data_set.class_indices)
test_data_set = test_datagen.flow_from_directory("/Users/user/Desktop/Samsung:Nokia/testset",
                                                  target_size=(64,64),
                                                  batch_size=50,
                                                  class_mode="binary")

model.fit_generator(train_data_set,
                         samples_per_epoch = 5000,
                         nb_epoch = 10,
                         validation_data = test_data_set,
                         nb_val_samples = 50)

#save model

model_json = model.to_json()
with open("model.json","w") as jsonfile:
    jsonfile.write(model_json)
    jsonfile.close()
model_weights = model.save_weights("model_weights.h5")
