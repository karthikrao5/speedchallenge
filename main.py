import cv2
import os 
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
# cap = cv2.VideoCapture("data/train.mp4")



# # Check if camera opened successfully
# if not cap.isOpened():
#     print("Error opening video stream or file")

# # Read until video is completed
# while cap.isOpened():
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if ret:
        
#         # edges = cv2.Canny(frame, 20, 200)

#         # Display the resulting frame
#         cv2.imshow('Frame', frame)

#         # Press Q on keyboard to  exit
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break

#     # Break the loop
#     else:
#         break
# # When everything done, release the video capture object 
# cap.release()

# # Closes all the frames
# cv2.destroyAllWindows()

images = []

if (not os.path.exists("trainMp4.npy")):


    cap = cv2.VideoCapture("data/train.mp4")

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    while True:
        flag, frame = cap.read() # get the frame
        if flag:
            # The frame is ready and already captured
            # cv2.imshow('video', frame)

            # store the current frame in as a numpy array
            image = cv2.cvtColor(cv2.resize(frame, (100, 100)) , cv2.COLOR_BGR2GRAY)
            images.append(image)
            # cv2.imshow("Frame", image)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            # pos_frame = cap.get(cv2.cv.CAP_PROP_POS_FRAMES)
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.cv.CAP_PROP_POS_FRAMES, pos_frame-1)
            print ("frame is not ready")
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)

        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break

    images = np.array(images)

    np.save('trainMp4.npy', images)
else:
    images = np.load('trainMp4.npy')
    


print("total images length: %d" % (len(images)))

train_num = int(len(images) * 0.8)
print ("train set length: %d" % (train_num))

train_images, test_images = images[:train_num,:], images[train_num:,:]

train_images = train_images / 255.0
test_images = test_images / 255.0

print("train images: %d, test images: %d" % (len(train_images), len(test_images)))

labels = []

with open('data/train.txt', 'r') as f:
    for line in f:
        labels.append(float(line))

print("label length: %d" % (len(labels)))
print("train labels length: %d" % (train_num))
print("Train images shape: %s" % str(train_images.shape))


train_labels = labels[:16320]
test_labels = labels[16320:]

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(100, 100)),
    keras.layers.Dense(128, activation='tanh'),
    keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0),
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=100)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)