# Hand-Recognition
> Hand Recognition project using OpenCV and Mediapipe framework provided by Google.

## Introduction

Hand gesture recognition is of great importance for human computer interaction (HCI) because of its extensive applications in virtual reality and sign language recognition etc. Human hand is very small and comparatively has complex articulations than any organ  in the entire human body. Gesture recognition is an active research field in Human-Computer Interaction technology (Computer Vision ). 

It has many applications in the virtual environment:
-	Sign language translation 
-	Robot control 
-	Music creation etc…

In this machine learning project on Hand Gesture Recognition, we are going to make a real-time Hand Gesture Recognizer using the MediaPipe framework in OpenCV and Python.

## Software Requirements

1.	Python 3.8.8
2.	OpenCV 4.5
3.	MediaPipe 0.8.5
4.	Numpy 1.19.3

## Data Flow Diagram ( for HandRecognition-FingerCounter )

![image](https://user-images.githubusercontent.com/112194179/187031033-067c8c3e-f49b-4eef-a5ab-8e507246cd9b.png)

## Approach
- Import required modules

- Initialize Webcam object 

- Read series of images captured from the WebCam

- Initialize the Mediapipe objects to recognise and map over Hands

- Traverse through the image folder to overlay images

- Convert landmark positions to accurate pixel values

- Compute whether a finger is open or closed based on landmark positions

- Embedd FPS functionality on the WebCam images

- Display the culminated output on screen

## References

- [OpenCV-Python and Mediapipe](https://www.youtube.com/watch?v=01sAkU_NvOY) - Code with OpenCV and Mediapipe  
