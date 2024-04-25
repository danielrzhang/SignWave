import mediapipe as mp
import cv2
import os
import pickle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIRECTORY = "./data"

data = []
labels = []

for directory in os.listdir(DATA_DIRECTORY):
    for image_path in os.listdir(os.path.join(DATA_DIRECTORY, directory)):
        data_auxiliary = []

        image = cv2.imread(os.path.join(DATA_DIRECTORY, directory, image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_auxiliary.append(x)
                    data_auxiliary.append(y)

                data.append(data_auxiliary)
                labels.append(directory)

f = open("data.pickle", "wb")
pickle.dump({"data": data, "labels": labels}, f)
f.close()
