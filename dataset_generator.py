# Generates image datasets for training

import mediapipe as mp
import cv2
import os
import pickle

# Initializing MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory where the image datasets are stored
DATA_DIRECTORY = "./data"

# Lists to store the extracted data (hand landmarks) and their corresponding labels
data = []
labels = []

# Iterating through each directory (class) in the data directory
for directory in os.listdir(DATA_DIRECTORY):

    # Iterating through each image in the current directory
    for image_path in os.listdir(os.path.join(DATA_DIRECTORY, directory)):

        # Temporary list to store landmarks for each image
        data_auxiliary = []

        # Reading the image
        image = cv2.imread(os.path.join(DATA_DIRECTORY, directory, image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Processing the image with the Hands model
        results = hands.process(image_rgb)

        # Checking if hands were detected in the image
        if results.multi_hand_landmarks:

            # Iterating through each detected hand
            for hand_landmarks in results.multi_hand_landmarks:

                # Iterating through each landmark point in the hand
                for i in range(len(hand_landmarks.landmark)):

                    # Extracting the x and y coordinates of each landmark point and appending the coordinates to a temporary list
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_auxiliary.append(x)
                    data_auxiliary.append(y)

                # Appending the list of landmark coordinates to the main data list and the corresponding label
                data.append(data_auxiliary)
                labels.append(directory)

# Saving the data and labels to a pickle file for later use
f = open("data.pickle", "wb")
pickle.dump({"data": data, "labels": labels}, f)
f.close()
