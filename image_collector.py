# Python script to take ASL (American Sign Language) character training images from the camera

import os
import cv2

# Directory to store the training images
DATA_DIR = './data'

# Creates the "data" directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of classes (ASL characters)
number_of_classes = 26

# Size of the dataset for each class
dataset_size = 100

# Opens the camera for video capturing
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Creates the directories each ASL character to store training images
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    # Prints the class being processed
    print('Collecting data for class {}'.format(j))

    # Loop to capture training images for each class
    done = False
    while True:
        # Reads a frame from the camera
        ret, frame = cap.read()

        # Displays instructions on the frame
        cv2.putText(frame, 'Press the space bar to take training images', (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Waits for the space bar to be pressed to proceed
        if cv2.waitKey(25) == ord(' '):
            break

    # Captures 100 training images for each class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        # Saves the captured image in the respective class folder
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
        counter += 1

# Releases the camera and closes all OpenCV windows
cap.release()
cv2.destroyAllWindows()