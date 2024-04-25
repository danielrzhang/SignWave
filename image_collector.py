# Python script to take ASL character training images from the camera

import os
import cv2

DATA_DIR = './data'

# Creates the "data" directory to store the folders with training images
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
dataset_size = 100

# Opens the camera for video capturing
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Creates the directories for the training images for each character
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False

    # Opens camera GUI on-screen
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Press the space bar to take training images', (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Waits for space bar action
        if cv2.waitKey(25) == ord(' '):
            break

    # Takes 100 training images in rapid succession
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()