## Import Modules
import csv # to read driver_log csv files
import cv2 # for image reading and processsing
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout 
from keras.layers.convolutional import Conv2D
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint 

# Read the driving log csv files from the two tracks
lines = []
data_files = ['driving_log_track1.csv', 'driving_log_track_2.csv',  'driving_log_zigzag.csv', 'driving_log_udacity.csv']

for file in data_files:
    csvfile = open(file)
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
    csvfile.close() 
    
# Using Scikit train-test-Split divide the data files into training (80%) and validation set (20%)
train_samples, validation_samples = train_test_split(lines, test_size=0.2, random_state=45)


# Visualizing the data and Cropping the images
def distort_image(img, angle, rot = 30, shift_px = 10):
    """
    Function to introduce random distortion: brightness, flip, rotation, and shift 
    """
    rows, cols,_ = img.shape
    choice = np.random.randint(5)
    #print(choice)
    if choice == 0:  # Randomly rotate 0-30 degreee
        rot *= np.random.random()   
        M = cv2.getRotationMatrix2D((cols/2,rows/2), rot, 1)
        dst = cv2.warpAffine(img,M,(cols,rows))
    elif choice == 1: # Randomly change the intensity
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
        hsv[:, :, 2] = hsv[:, :, 2] * ratio
        dst = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    elif choice == 2: # Randomly shift the image in horizontal and vertical direction
        x_shift,y_shift = np.random.randint(-shift_px,shift_px,2)
        M = np.float32([[1,0,x_shift],[0,1,y_shift]])
        dst = cv2.warpAffine(img,M,(cols,rows))
    elif choice == 3: # Randomly flip the image
        dst = np.fliplr(img)
        angle = (-1.0) * float(angle)
    else:
        dst = img
    return dst, float(angle)


# data generator
def data_generator(samples, batch_size=32, corr=0.35, validation_flag = False):
    """
    Function to generate data after, it reads the image files, performs random distortions and finally 
	returns a batch of training or validation data
    """
    num_samples = len(samples)
    correction = [0, corr, -corr]
    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                if validation_flag:  # The validation data consists only of center image and without distortions
                    name = 'IMG/' + batch_sample[0].split('/')[-1]
                    image = cv2.imread(name)
                    angle = float(batch_sample[3]) 
                    images.append(image)
                    angles.append(angle)
                    continue
                else:  # In training dataset we introduce distortions to augment it and improve performance
                    for index in range(3):  # Add all center, left and right images to the training dataset
                        name = 'IMG/' + batch_sample[index].split('/')[-1]
                        angle = float(batch_sample[3]) + float(correction[index])
                        image = cv2.imread(name)
                        # Randomly augment the training dataset to reduce overfitting
                        image, angle = distort_image(image, angle)
                        images.append(image)
                        angles.append(angle)

            # Convert the data into numpy arrays
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)  



if __name__ == '__main__':
    # define the training and validation data generators
    train_generator = data_generator(train_samples, batch_size=32)
    validation_generator = data_generator(validation_samples, batch_size=32, validation_flag=True)
    input_shape = (160,320,3)

    ## Create a CNN model mbased on NVIDIA's DAVE-2 model (https://arxiv.org/pdf/1604.07316v1.pdf)
    model = Sequential()

    # Hard wired Cropping layer to crop the image and send as input only the relevant Region of Interest
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=input_shape))

    # Hardwired Normalization layer to normalize the input
    model.add(Lambda(lambda x: x/127.5 - 1.0))  # Normalize the input and center the mean.

    # Three convilutional layers with Kernel 5x5
    model.add(Conv2D(24, (5,5), strides=(2, 2), padding='valid', activation='elu', use_bias=True))
    model.add(Conv2D(36, (5,5), strides=(2, 2), padding='valid', activation='elu', use_bias=True))
    model.add(Conv2D(48, (5,5), strides=(2, 2), padding='valid', activation='elu', use_bias=True))

    # Two Convolutional layers with Kernel 3x3
    model.add(Conv2D(64, (3,3), strides=(1, 1), padding='valid', activation='elu', use_bias=True))
    model.add(Conv2D(64, (3,3), strides=(1, 1), padding='valid', activation='elu', use_bias=True))

    # A droput layer to reduce overfitting
    model.add(Dropout(0.5))

    # Flatten the output of last convolutional layer so that it can be fed to MLP layers
    model.add(Flatten())

    # Four Fully connected layers to predict steering angle from the input features
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    # define the mean square error loss and Adam optimizer
    model.compile(loss='mse', optimizer='adam')

    # Create a checkpointer so that we save only the best model
    checkpointer = ModelCheckpoint(filepath='model.h5', monitor='val_loss', verbose=1, save_best_only=True)

    # Train the model for with taining and validation dataset generated using generators
    # To ensure best performance the best model is only saved using CheckPointer
    history = model.fit_generator(train_generator, steps_per_epoch = 1000, validation_data=validation_generator,
                              validation_steps= 500, epochs=30, verbose=1, callbacks=[checkpointer])

    # Plot the training and validation losses
    plt.plot(history.history['loss'][1:])
    plt.plot(history.history['val_loss'][1:])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'Validation'], loc='upper left')
    plt.show()


