import csv
from scipy import ndimage
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D
from keras.layers.convolutional import Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda, Dropout
from keras.callbacks import ModelCheckpoint

lines = []

with open('sample_training_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines[1:]:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'sample_training_data/IMG/' + filename

    image = ndimage.imread(current_path, mode='RGB')
    images.append(image)
    measurements.append(float(line[3]))

    # Add mirrored image
    image_flipped = np.fliplr(image)
    images.append(image_flipped)
    measurements.append(-float(line[3]))


X_train = np.array(images)
y_train = np.array(measurements)
rate = 0.5

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
#model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(BatchNormalization())
model.add(Conv2D(8, (1, 1), activation="relu"))
model.add(Conv2D(16, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(32, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation="relu", strides=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dropout(rate))
model.add(BatchNormalization())
model.add(Dense(512))
model.add(Dropout(rate))
model.add(BatchNormalization())
model.add(Dense(256))
model.add(Dropout(rate))
model.add(Dense(64))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train, validation_split=0.3, shuffle=True, epochs=50, verbose = 1)

model.save('model.h5')
