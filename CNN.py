from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initializing the CNN
clf = Sequential()

#Step 1 - Convolution layer
clf.add(Convolution2D(32, (3,3), activation = 'relu'))

#Step 2 - MaxPooling layer
clf.add(MaxPooling2D(pool_size = (2,2)))

#Step 3 - Flattening layer
clf.add(Flatten())

#Step 4 - Full Connection layer
clf.add(Dense(units  = 128, activation = 'relu'))
clf.add(Dense(units = 1, activation = 'sigmoid'))

#compiling the CNN
clf.compile('adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')


clf.fit_generator(training_set, 
                    steps_per_epoch=8000,
                    epochs=25,
                    validation_data = test_set,
                    validation_steps=2000)
