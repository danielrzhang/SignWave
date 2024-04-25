# Main program for detecting live ASL hand gestures

import cv2
import mediapipe as mp
import pickle
import numpy as np

# Loading the trained model from the pickle file
model_dictionary = pickle.load(open("./model.p", "rb"))
model = model_dictionary["model"]

# Dictionary to map predicted labels to characters (A-Z)
labels_dictionary = {i: chr(65 + i) for i in range(25)}

# Initializing the video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Initializing MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Continuous loop to process frames from the webcam
while True:
    # Lists to store hand landmarks
    data_auxiliary = []
    xPositions = []
    yPositions = []

    # Reading a frame from the camera
    ret, frame = cap.read()
    HEIGHT, WIDTH, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processing the frame with the Hands model
    results = hands.process(frame_rgb)

    # Checking if hand landmarks are detected and if only one hand is detected
    landmarks = results.multi_hand_landmarks
    if landmarks and len(landmarks) <= 1:

        # Draws landmarks and connections on the frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Extracts x and y coordinates of hand landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_auxiliary.append(x)
                data_auxiliary.append(y)
                xPositions.append(x)
                yPositions.append(y)

        # Calculates bounding box coordinates for the hand region
        x1 = int(min(xPositions) * WIDTH) - 10
        y1 = int(min(yPositions) * HEIGHT) - 10
        x2 = int(max(xPositions) * WIDTH) - 10
        y2 = int(max(yPositions) * HEIGHT) - 10

        # Makes a character prediction using the trained model
        prediction = model.predict([np.asarray(data_auxiliary)])
        predicted_character = labels_dictionary[int(prediction[0])]

        # Drawing bounding box and predicted character on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Displays the frame
    cv2.imshow("SignWave", frame)
    cv2.waitKey(1)
