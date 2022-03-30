# Team-22053_AdaptiveRobotGripper
repository to hold all software for ENGR498 Team-22053

## Raspberry Pi Setup
The Raspberry Pi is an Operating System
You will need
* a keyboard
* a mouse
* a small-HDMI compatible display
Insert the microSD card into a SD card to USB adapter, and then plug the adapter into your computer. Download and install Raspberry Pi 64 bit Bullseye.
https://www.raspberrypi.com/documentation/computers/getting-started.html
The default username: pi and password: raspberry

## VNC Viewer
VNC Viewer is a great tool to remotely access the Raspberry Pi. Download it on your computer, and then connect via username/password, as well as the IP address. You can then transfer files to and from the Pi, as well as run software remotely.


## Object Recognition Software

Object recognition software was trained using yolov5. Raspberry Pi 64-bit can only access the camera via C++, but Raspberry Pi 32-bit is unable to import certain libraries from the object recognition software, so only 64-bit can do object recognition.

Steps to train image recognition software 
1. Download Visual Studio Code, and git pull the yolov5 repository: https://github.com/ultralytics/yolov5
2. Download Python, pip, and all required libraries
3. Gather images of product and annotate using https://roboflow.com, we want to use images oriented similar to final conveyer system under similar lighting to speed up and ensure accuracy of the AI 
4. Export these images to yolov5 formatted zip folder
5. Configure yolov5 files to train on our custom dataset, saving the CNN as a .pt file
6. Export .pt file as a .onnx file for deployment on Raspberry Pi

Steps to deploy image recognition software on Raspberry Pi
1. Install Bullseye 64-bit OS on Raspberry Pi
2. Git pull the C++ yolov5 port: https://github.com/doleron/yolov5-opencv-cpp-python 
3. scp the .onnx file over to the Raspberry Pi
4. Run camera software and image recognition software
    The output is currently a video of the bounding box recognition on the camera input
    Use the python command python python/yolo.py to run the program
    Updated yolounilever.py displays the results classifications with confidences, without the video, and takes live images

https://www.raspberrypi.com/documentation/accessories/camera.html

## Object location software

https://tutorials-raspberrypi.com/raspberry-pi-ultrasonic-sensor-hc-sr04/

1. Set up the HC SR-04 ultrasonic sensor described in the TDP
2. Run distance_sensor.py
    The output is in cm

## Force sensor software

https://pimylifeup.com/raspberry-pi-pressure-pad/

1. Set up the FSR sensor described in the TDP
2. Run force_sensor.py
    The output is currently in the range from 0-1023, working on converting this digital value to a force

## Stepper Motor software

currently WIP
