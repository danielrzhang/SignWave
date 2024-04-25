# Trains the classifier to recognize ASL hand images

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Loading the data and labels from the pickle file
data_dictionary = pickle.load(open("./data.pickle", "rb"))

# Converting data and labels to numpy arrays
data = np.asarray(data_dictionary["data"])
labels = np.asarray(data_dictionary["labels"])

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Training the classifier on the training data to make predictions and calculate the accuracy score
model = RandomForestClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

# Saving the trained model to a pickle file
f = open("model.p", "wb")
pickle.dump({"model": model}, f)
f.close()


