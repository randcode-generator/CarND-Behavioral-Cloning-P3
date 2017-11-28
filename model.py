import csv
from PIL import Image
import numpy as np

listOfFiles = []

"""
Load path to image file in driving_log.csv
Args:
  path: path to directory that contains driving_log.csv
  nudgeBy: adjustments to the steering angle
"""
def processFile(path, nudgeBy):
  with open(path + "driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
      #read the file path to center camera
      source_path = line[0]
      #get the filename
      filename = source_path.split('/')[-1]
      #complete path
      #data-track1-full//IMG/center_2017_11_06_15_04_00_606.jpg
      completePath = path + '/IMG/'+ filename
      steeringAngle = float(line[3]) + nudgeBy
      #add the path and steering angle as tuple
      listOfFiles.append((completePath, steeringAngle))

#process image file and steering angle adjustments
#repeatly add the same images and angle to get the steering angles you are looking for
for i in range(1):
  processFile('data-track1-full/', 0)

for i in range(1):
  processFile('data-track1-turn1/', -0.5)

for i in range(2):
  processFile('data-track1-turn2/', -0.5)

for i in range(30):
  processFile('data-track1-turn3/', 0.5)

for i in range(3):
  processFile('data-track2-full/', 0)

for i in range(6):
  processFile('data-track2-turn2/', 1.0)

for i in range(6):
  processFile('data-track2-turn3/', -1.0)

for i in range(6):
  processFile("data-track2-turn4/", 0)

for i in range(285):
  processFile("data-track2-turn5/", 1.0)

for i in range(30):
  processFile("data-track2-turn6/", -0.1)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(listOfFiles, test_size=0.2)

from random import shuffle
import sklearn

def generator(samples, batch_size=32):
  num_samples = len(samples)
  while 1:
    shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]

      images = []
      angles = []
      for batch_sample in batch_samples:
        image = Image.open(batch_sample[0])
        images.append(np.array(image))
        angles.append(batch_sample[1])

      X_train = np.array(images)
      y_train = np.array(angles)

      yield sklearn.utils.shuffle(X_train, y_train)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#Reuse model if another model exists. Otherwise, create a new model.
from pathlib import Path
model_file = Path("model_really_good.h5")
if model_file.is_file():
  from keras.models import load_model
  model = load_model('model_really_good.h5')
else:
  model = Sequential()
  model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
  model.add(Lambda(lambda x: (x / 255.0) - 0.5))
  model.add(Convolution2D(6,5,5,subsample=(1,1),activation="relu"))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))
  model.add(Convolution2D(16,5,5,subsample=(1,1),activation="relu"))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))
  model.add(Flatten())
  model.add(Dense(200,activation="sigmoid"))
  model.add(Dense(100,activation="sigmoid"))
  model.add(Dense(1))

#save the model for every epoch.
filepath="model-{epoch:02d}-{val_loss:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
callbacks_list = [checkpoint]

model.compile(loss='mse', optimizer='adam')

#use generator so it consumes less memory
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10, callbacks=callbacks_list)
