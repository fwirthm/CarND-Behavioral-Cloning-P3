#import
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, AveragePooling2D, Cropping2D
from keras.callbacks import EarlyStopping
from scipy import ndimage

#define correction factor for left and right images
correction = 0.2

#set up lists for training images and measurements (steering angles = outputs)
images = []
measurements = []

#load all images which were 
for folder in ['udacity_training', 'my_data_track1', 'my_data_track2', 'borders', 'more_borders']:
    lines = []
    with open('./data/'+folder+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
     
    for lineid, line in enumerate(lines[1:]):
        meas = float(line[3])
        #if the steering angle is zeor, this is an trivial example, therefore
        ##### - only a smal portion of these images are used
        ##### - the image is not duplicated through flipping or using left and right camera images
        if meas == 0.0:
            if lineid%100==0:
                measurements.append(meas)
                source = line[0].split('/')[-1]
                image = ndimage.imread('./data/'+folder+'/IMG/'+source, mode='RGB')
                images.append(image)
            else:
                pass
        else:
            #for all interesting examples
            #the images of all three cameras as well as its flipped variants are added to the images list
            #steering angles corresponding to these images are added to the measurements list - for the left and right images a correction is applied
            meas_flip = -meas

            measurements.append(meas)
            measurements.append(meas_flip)

            measurements.append(meas+correction)
            measurements.append(meas_flip-correction)

            measurements.append(meas-correction)
            measurements.append(meas_flip+correction)

            for i in range(3):
                source = line[i].split('/')[-1]
                image = ndimage.imread('./data/'+folder+'/IMG/'+source, mode='RGB')
                images.append(image)

                image_flip = np.fliplr(image)
                images.append(image_flip)

#convert lists to arrays
X_train = np.array(images)
y_train = np.array(measurements)

#implement network architecture presented by NVIDIA for its imitation learning project
model = Sequential()
model.add(Lambda(lambda x: (x/255.)-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))

model.add(Conv2D(filters=24, kernel_size=(5, 5), activation='relu', strides=(2,2)))
model.add(Conv2D(filters=36, kernel_size=(5, 5), activation='relu', strides=(2,2)))
model.add(Conv2D(filters=48, kernel_size=(5, 5), activation='relu', strides=(2,2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(Flatten())

model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=10, activation = 'relu'))
model.add(Dense(units=1))

model.compile(loss='mse', optimizer='adam')

#fit the model with early stopping
callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, callbacks=[callback], epochs=10)

#save the model
model.save('model.h5')
