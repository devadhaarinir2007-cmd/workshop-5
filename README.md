# workshop-5-License Plate Detection using OpenCV and Haar Cascade Classifier
License Plate Detection using OpenCV and Haar Cascade Classifier Name: vinodhini Register Number: 212225230305

## Aim
To implement a License Plate Detection system using OpenCV and Haar Cascade Classifier, draw bounding boxes, crop the detected region, and blur the license plate to improve privacy. The detection accuracy is improved by tuning Haar Cascade parameters.

## Software Used
Python 3.7 or above OpenCV (opencv-python) NumPy Matplotlib Jupyter Notebook (Anaconda) Haar Cascade File: haarcascade_russian_plate_number.xml

## Algorithm
Import necessary libraries such as OpenCV and Matplotlib Read the input vehicle image Convert the original image to grayscale for faster computation Load the Haar Cascade classifier for license plate detection Detect license plate using detectMultiScale function Draw rectangle around detected area Crop the detected region using numpy slicing with (x, y, w, h) values Apply median blurring on the cropped region Replace the original region with blurred version Display final result using Matplotlib

## Program:
```
import cv2
import matplotlib.pyplot as plt

def display(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,6))
    plt.imshow(img_rgb)
    plt.axis('off')

img = cv2.imread("DATA/car_plate.jpg")
plate_cascade = cv2.CascadeClassifier("DATA/haarcascades/haarcascade_russian_plate_number.xml")

def detect_and_blur_plate(img):
    img_copy = img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in plates:
        roi = img_copy[y:y+h, x:x+w]
        blurred_roi = cv2.medianBlur(roi, 15)
        img_copy[y:y+h, x:x+w] = blurred_roi

    return img_copy

result = detect_and_blur_plate(img)
display(result)
```
## OUTPUT:
<img width="1043" height="597" alt="image" src="https://github.com/user-attachments/assets/b3041df2-c8d8-412d-b2c1-ba440d5499fc" />
<img width="1050" height="592" alt="image" src="https://github.com/user-attachments/assets/49bda3e0-5511-40a3-b35a-71ead7b983b8" />
<img width="1048" height="607" alt="image" src="https://github.com/user-attachments/assets/19237249-3cde-46dd-8023-c0445f080593" />
