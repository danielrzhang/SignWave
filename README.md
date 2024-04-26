# SignWave

## Description
SignWave is designed to detect American Sign Language (ASL) hand gestures in real-time via webcam. It includes functionalities to collect training images for ASL characters from the camera, generate image datasets for training, train a classifier to recognize ASL hand images, and perform live detection of ASL characters. This program was developed using Python, OpenCV, MediaPipe, and scikit-learn.

## Files
1. `image_collector.py`
   * Python script to capture ASL character training images from the camera
2. `dataset_generator.py`
    * Script to generate image datasets for training from collected images
3. `image_classifier.py`
    * Trains the classifier to recognize ASL hand images
4. `main.py`
    * Main program for detecting live ASL hand gestures

## Usage
1. **Image Collection**
   * Run `image_collector.py` to collect ASL character training images from the camera. Images are saved in corresponding directories in the `./data` directory.
2. **Dataset Generation**
   * Execute `dataset_generator.py` to generate image datasets for training from the collected images. This dataset will be saved in `data.pickle`.
3. **Training**
   *  Run `image_classifier.py` to train the classifier on the generated datasets. The trained model is saved as `model.p`.
4. **Live Detection**
   * Execute `main.py` to detect ASL hand gestures in real-time using the webcam.

## Requirements
* Python 3.xx
* OpenCV (`pip install opencv-python`)
* MediaPipe (`pip install mediapipe`)
* scikit-learn (`pip install scikit-learn`)