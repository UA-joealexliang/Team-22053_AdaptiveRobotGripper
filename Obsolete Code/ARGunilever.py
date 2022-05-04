##############################################################################################################
#GUI Software
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk,Image
import numpy as np
import matplotlib.pyplot as plt
import os
from time import sleep 
from threading import Thread

border_effects = {

    "flat": tk.FLAT,

    "sunken": tk.SUNKEN,

    "raised": tk.RAISED,

    "groove": tk.GROOVE,

    "ridge": tk.RIDGE,

}

#sets up window
window = tk.Tk()
window.title('ARG GUI - 22053')
window.geometry("1920x1080")

#TEMP VARIBLE SETTING
is_new_image = False
path_to_img = ' test.jpg'
img_resize = (320,320)
PressureB = None
PressureN = None
Actual_Pressure = None
Guess = None
Confidence = None
Object = None #do not modify

"""
The Def sections below will be used to create the commands to run the seperate files such as the distant sensor, Camera system, object identification, ETC.
"""
#command to distroy the GUI window
def end():
    window.destroy()

#Takes whatever image from folder as output from camera system
PIL_image = Image.open(path_to_img)
resize_image = PIL_image.resize(img_resize)
img = ImageTk.PhotoImage(resize_image)
Scan = tk.Label(image=img)

#This outlines the array for steps and their values
Overview = [[0 for j in range (2)] for i in range (7)]
Overview[0][0]="Object Detection"
Overview[1][0]="Object Identified"
Overview[2][0]="Object Located"
Overview[3][0]="Object Grabbed"
Overview[4][0]="Object Moving"
Overview[5][0]="Object Dropped"
Overview[6][0]="Cycle Complete"
#Sets column 2 all values to 0
for i in range(0,7):
    Overview[i][1] = 0

#If statement for object confidence
def ConCheck():
    #TEMP you can set confidence to what ever output from image is
    #Confidence = 1
    #TEMP if confidence is above .7 we can retreve the guess from image output
    if Confidence == None:
        Object = "Object Identity: Waiting for Data"
    elif Confidence > .7:
        Object = "The object is " + Guess + " with confidence of " + str(Confidence)
    else:
        Object = "The object could not be Identified"

    Identity_value.config(text = Object)

def PressUpdate():
    #TEMP change to pressure from sensor
    if (PressureB == None or PressureN == None):
        Pressure_Value.config(text = "Pressure Value: Waiting for Data")
    else:
        Pressure_Value.config(text = str(PressureB) + " bits" + str(PressureN) + " N")


#This section defines all labels for steps of ARG
Object_label = tk.Label(window, text = Overview[0][0], font=('calibre',10, 'bold'), relief="raised", width=50)
Identified_label = tk.Label(window, text = Overview[1][0], font=('calibre',10, 'bold'), relief="raised", width=50)
Located_label = tk.Label(window, text =  Overview[2][0], font=('calibre',10, 'bold'), relief="raised", width=50)
Grabbed_label = tk.Label(window, text =  Overview[3][0], font=('calibre',10, 'bold'), relief="raised", width=50)
Moving_label = tk.Label(window, text =  Overview[4][0], font=('calibre',10, 'bold'), relief="raised", width=50)
Dropped_label = tk.Label(window, text =  Overview[5][0], font=('calibre',10, 'bold'), relief="raised", width=50)
Done_label = tk.Label(window, text =  Overview[6][0], font=('calibre',10, 'bold'), relief="raised", width=50)
Camera_label = tk.Label(window, text = "Camera System", font=('calibre',10, 'bold'), relief="raised", width=50)
Pressure_label = tk.Label(window, text = "Pressure System", font=('calibre',10, 'bold'), relief="raised", width=50)
Identity_label = tk.Label(window, text = "Object Identity", font=('calibre',10, 'bold'), relief="raised", width=50)
#This sets up the frames for the status system
#manually lays out the 7 steps on the grid
Object_label.grid(row=0,column=0, sticky="w")
Identified_label.grid(row=1,column=0, sticky="w")
Located_label.grid(row=2,column=0, sticky="w")
Grabbed_label.grid(row=3,column=0, sticky="w")
Moving_label.grid(row=4,column=0, sticky="w")
Dropped_label.grid(row=5,column=0, sticky="w")
Done_label.grid(row=6,column=0, sticky="w")

#Prototype of the status system using numbers
Object_value = tk.Label(window, width=10, bg="red")
Identified_value = tk.Label(window, width=10, bg="red")
Located_value = tk.Label(window, width=10, bg="red")
Grabbed_value = tk.Label(window, width=10, bg="red")
Moving_value = tk.Label(window, width=10, bg="red")
Dropped_value = tk.Label(window, width=10, bg="red")
Done_value = tk.Label(window, width=10, bg="red")
Pressure_Value = tk.Label(window, text = "Pressure Value: Waiting for Data" , font=('calibre',10, 'bold'), relief="raised", width=50)
Identity_value = tk.Label(window, text = "Object Identity: Waiting for Data", font=('calibre',10, 'bold'), relief="raised", width=50)
P_v = [0 for i in range (9)]
P_v[0] = Object_value
P_v[1] = Identified_value
P_v[2] = Located_value
P_v[3] = Grabbed_value
P_v[4] = Moving_value
P_v[5] = Dropped_value
P_v[6] = Done_value
P_v[7] = Pressure_Value
P_v[8] = Identity_value
#Manually lays out all 7 values
Object_value.grid(row=0,column=1, sticky="w")
Identified_value.grid(row=1,column=1, sticky="w")
Located_value.grid(row=2,column=1, sticky="w")
Grabbed_value.grid(row=3,column=1, sticky="w")
Moving_value.grid(row=4,column=1, sticky="w")
Dropped_value.grid(row=5,column=1, sticky="w")
Done_value.grid(row=6,column=1, sticky="w")

#Creates command to change status values which should lead into color changes if possible
def set(n, val):
    Overview[n][1] = val

def check(n):
    if Overview[n][1] == 0:
        P_v[n].config(bg="red")
    elif Overview[n][1] == 1:
        P_v[n].config(bg="yellow")
    elif Overview[n][1] == 2:
        P_v[n].config(bg="green")

def reset():
    global PressureB, PressureN, Actual_Pressure, Guess, Confidence
    PressureB = None
    PressureN = None
    Actual_Pressure = None
    Guess = None
    Confidence = None
    #ConCheck()
    #PressUpdate()
    for i in range(0,7):
        Overview[i][1] = 0
        #P_v[i].config(bg="red")
        print("red")

exit_button = ttk.Button(window, text="Exit", command=end)
reset_button = ttk.Button(window, text="Reset", command=reset)

#Displays label and image from camera system
Camera_label.grid(row=0,column=3, sticky="s")
Scan.grid(row=1,column=3, rowspan = 6)

#Displays Pressure values and label
Pressure_label.grid(row=10,column=3)
Pressure_Value.grid(row=11,column=3)

#Displays Object idenity and confidence
Identity_label.grid(row=10,column=0)
Identity_value.grid(row=11,column=0)

#Displays graph and exit button
exit_button.grid(row=13)
reset_button.grid(row=12)

def GUI_Thread():
    while True:
        window.update()
        window.update_idletasks()
        ConCheck()
        PressUpdate()
        for i in range(0,7):
            check(i)
        if is_new_image == True:
            is_new_image = False
            PIL_image = Image.open(path_to_img)
            resize_image = PIL_image.resize(img_resize)
            img = ImageTk.PhotoImage(resize_image)
            Scan = tk.Label(image=img)
            Scan.grid(row=1,column=3, rowspan = 6)

guiThread = Thread(target = GUI_Thread)
guiThread.start()
################################################################################################ 

################################################################################################
#Button
import threading, sys, os

class MyThread ( threading.Thread ):

   def run ( self ):
      print('Hello and good bye.')
      os._exit(1)

import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
i = 0
def button_callback(channel):
    global i
    i = i + 1
    print("Button was pushed! ",i)
    MyThread().start()
GPIO.setwarnings(False) # Ignore warning for now
GPIO.setmode(GPIO.BCM) # Use physical pin numbering
GPIO.setup(2, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 10 to be an input pin and set initial value to be pulled low (off)
GPIO.add_event_detect(2,GPIO.RISING,callback=button_callback) # Setup event on pin 10 rising edge
################################################################################################

import cv2
import time
import sys
import numpy as np
import subprocess
import os
import time

#import libraries for stepper motor
from time import sleep

from sympy import false
#import RPi.GPIO as GPIO

#import libraries for distance sensor
import RPi.GPIO as GPIO 
import time

#import libraries for pressure sensor
import spidev
import time

#Stepper Motor Constants
DIR = 20 #Direction GPIO Pin
STEP = 21 #Step GPIO PIN
CW = 0 #CLOCKWISE ROTATION
CCW = 1 #CCW ROTATION
SPR = 800 #steps per revolution
GPIO.setmode(GPIO.BCM)
GPIO.setup(DIR,GPIO.OUT)
GPIO.setup(STEP,GPIO.OUT)
GPIO.output(DIR,CW)
MODE = (14,15,18) #MICROSTEP RESOLUTION GPIO
GPIO.setup(MODE, GPIO.OUT)
RESOLUTION = {'1':(0,0,0), #Full
              '2':(1,0,0), #Half
              '4':(0,1,0),  #1/4
              '8':(1,1,0),  #1/8
              '16':(0,0,1), #1/16
              '32':(1,0,1)} #1/32
#GPIO.output(MODE,RESOLUTION['1/32'])
#step_count = 200 * 32 #SPR
#delay = .0005 / 32

#Distance Sensor Constants
GET_DISTANCE_DELAY = 0.01 #delay between each distance sensor read
PRINT_DISTANCE = True
DISTANCE_SENSOR_LOADED = False
DISTANCE_THRESHOLD = 0.5

#Force Sensor Constants
GET_FORCE_DELAY = 0.01 #delay between each force sensor read
PRINT_FORCE = False
pad_channel = 0
measured_weight = 10
measured_bit = 1023
measured_n = measured_bit/(measured_weight*9.807) #max value is 1023 bits, corresponding to 10kg
#Create SPI
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz=1000000

def open_steps(steps=1600, delay=.005/16):
    print(steps)
    GPIO.output(MODE,RESOLUTION['16'])
    GPIO.output(DIR,CW)
    for x in range(steps):
            GPIO.output(STEP, GPIO.HIGH)
            sleep(delay)
            GPIO.output(STEP, GPIO.LOW)
            sleep(delay)
           
def close_steps(steps, delay):
    GPIO.output(DIR,CCW)
    for x in range(steps):
        GPIO.output(STEP, GPIO.HIGH)
        sleep(delay)
        GPIO.output(STEP, GPIO.LOW)
        sleep(delay)

def reset_close(steps,delay=0.005/16):
    GPIO.output(MODE,RESOLUTION['16'])
    GPIO.output(DIR,CCW)
    for x in range(steps):
        GPIO.output(STEP, GPIO.HIGH)
        sleep(delay)
        GPIO.output(STEP, GPIO.LOW)
        sleep(delay)    

def grip_init():
    INIT_RES = '16'
    GPIO.output(MODE,RESOLUTION[INIT_RES])
    INIT_OPEN_STEPCNT = 1600
    INIT_OPEN_DELAY = 0.0005
    open_steps(INIT_OPEN_STEPCNT, INIT_OPEN_DELAY/INIT_RES)

def grip_bodywash():
    BODYWASH_RES = '16'
    GPIO.output(MODE,RESOLUTION[BODYWASH_RES])
    BODYWASH_STEPCNT = 1600
    BODYWASH_DELAY = 0.005/16
    close_steps(BODYWASH_STEPCNT, BODYWASH_DELAY)
    close_steps(200,.075/16)
    return 400, 3000
    
def grip_deodorant():
    DEODORANT_RES = '16'
    GPIO.output(MODE,RESOLUTION[DEODORANT_RES])
    DEODORANT_STEPCNT = 1600
    DEODORANT_DELAY = 0.005/16
    close_steps(DEODORANT_STEPCNT, DEODORANT_DELAY)
    close_steps(1600,.005/16)
    return 400, 800
    
def grip_soap():
    SOAP_RES = '8'
    GPIO.output(MODE,RESOLUTION[SOAP_RES])
    SOAP_STEPCNT = 700
    SOAP_DELAY = 0.0005/8
    close_steps(SOAP_STEPCNT, SOAP_DELAY)
    #close_steps(100,.075/16)
    return 400, 800
    
def grip_spray():
    SPRAY_RES = '8'
    GPIO.output(MODE,RESOLUTION[SPRAY_RES])
    SPRAY_STEPCNT = 750
    SPRAY_DELAY = 0.0005/8
    close_steps(SPRAY_STEPCNT, SPRAY_DELAY)
    #close_steps(100,.075/8)
    return 400, 800
    
def grip_tea():
    TEA_RES = '16'
    GPIO.output(MODE,RESOLUTION[TEA_RES])
    TEA_STEPCNT = 1800
    TEA_DELAY = 0.005/16
    close_steps(TEA_STEPCNT, TEA_DELAY)
    return 400, 800

#https://thepihut.com/blogs/raspberry-pi-tutorials/hc-sr04-ultrasonic-range-sensor-on-the-raspberry-pi
def distance_sensor():
    #first time? set everything up
    global DISTANCE_SENSOR_LOADED
    if DISTANCE_SENSOR_LOADED == False:
        GPIO.setmode(GPIO.BOARD) #set reading pin to using board
        global PIN_TRIGGER
        PIN_TRIGGER = 7
        global PIN_ECHO
        PIN_ECHO = 11

        GPIO.setup(PIN_TRIGGER, GPIO.OUT) #out - triggers sensor
        GPIO.setup(PIN_ECHO, GPIO.IN) #in - reads return signal

        DISTANCE_SENSOR_LOADED = True
        print("DISTANCE_SENSOR_LOADED = True")

    #10us pulse to trigger the module
    GPIO.output(PIN_TRIGGER,GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(PIN_TRIGGER, GPIO.LOW)

    #echo pulse is low until called, then high for duration of pulse
    while GPIO.input(PIN_ECHO)==0:
        pulse_start_time = time.time()
    while GPIO.input(PIN_ECHO)==1:
        pulse_end_time = time.time()

    #math calculations
    pulse_duration = pulse_end_time - pulse_start_time
    distance = round(pulse_duration * 17150, 2)
    #alpha_distance = distance/2.54
    #print("Distance:",distance,"cm")
    #print("Distance:",alpha_distance,"in")
    return distance

def poll_distance_until(delay=GET_DISTANCE_DELAY, printD=PRINT_DISTANCE, threshold=DISTANCE_THRESHOLD):
    threshold_reached = False
    while threshold_reached==False:
        distance = distance_sensor()
        if printD==True:
            print("Distance:",distance,"cm")
        time.sleep(delay)
        if distance <= threshold:
            threshold_reached = True
            print("Object detected/located: ",distance,"cm")
            return distance

def poll_distance(delay=GET_DISTANCE_DELAY, printD=PRINT_DISTANCE):
    try:
        while True:
            distance = distance_sensor()
            if printD==True:
                print("Distance:",distance,"cm")
            time.sleep(delay)
    except KeyboardInterrupt:
        print("get_distance stopped by user")
        DISTANCE_SENSOR_LOADED = False
        print("DISTANCE_SENSOR_LOADED = False")
        GPIO.cleanup()
        
def readadc(adcnum):
    # read SPI data from the MCP3008, 8 channels in total
    if adcnum > 7 or adcnum < 0:
        return -1
    r = spi.xfer2([1, 8 + adcnum << 4, 0])
    data = ((r[1] & 3) << 8) + r[2]
    return data

def force_sensor():
    pad_value = readadc(pad_channel)
    measured_F = pad_value/measured_n
    return measured_F, pad_value

def poll_force2(delay=GET_FORCE_DELAY, print_force=PRINT_FORCE):
    while True:
        force, pad_value = force_sensor()
        if print_force==True:
            print("Pressure Pad Value: %d" % pad_value)
            #print("Pressure Pad Force: %f" % force)
        global PressureB
        global PressureN
        PressureB = pad_value
        PressureN = force
        return force, pad_value
        time.sleep(delay)

def poll_force(delay=GET_FORCE_DELAY, print_force=PRINT_FORCE):
    try:
        while True:
            force, pad_value = force_sensor()
            if print_force==True:
                print("Pressure Pad Value: %d" % pad_value)
                #print("Pressure Pad Force: %f" % force)
            global PressureB
            global PressureN
            PressureB = pad_value
            PressureN = force
            return force, pad_value
            time.sleep(delay)
    except KeyboardInterrupt:        
        pass

def remove_picture(path):
    if os.path.exists(path):
        os.remove(path)
    else:
        print("The file does not exist")

def take_picture():
    subprocess.run(["ls", "-l"])
    subprocess.run(["libcamera-jpeg","-o test.jpg", "-n", "-t", "1", "--width", "640", "--height", "640"])
#libcamera-jpeg -o test.jpg -t 2000 --width 640 --height 480
    #subprocess.run(["libcamera-jpeg","-o test.jpg", "-t 0", "--width 640", "--height 480"])
#read the onnx trained model
def build_model(is_cuda):
    net = cv2.dnn.readNet("config_files/bestexp16.onnx")
    if is_cuda:
        print("Attempty to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

#constants for image
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4

def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds

#load capture, returns cv2::Video
def load_capture():
    capture = cv2.VideoCapture("sample.mp4")
    return capture

def load_img(path):
    img = cv2.imread(path)
    return img

#load Unilever classes
def load_classes():
    class_list = []
    with open("config_files/unileverclasses.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list

class_list = load_classes()

#for input_image, returns result_class_ids, result_confidences, result_boxes
def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.7:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .7):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes

#format frame
def format_yolov5(frame):

    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result

#colors of bounding boxes
colors = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255)]

is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"

net = build_model(is_cuda)

def detect_img(path, start):
    
    frame = load_img(" test.jpg")

    #_, frame = img.read()
    inputImage = format_yolov5(frame)
    outs = detect(inputImage, net)
    class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])
    
    end = time.time()
    for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            color = colors[int(classid) % len(colors)]
            print(classid, class_list[classid], confidence, color)
    print("time elapsed:", end-start)
    
    for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            color = colors[int(classid) % len(colors)]
            cv2.rectangle(frame, box, color, 2)
            cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv2.putText(frame, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            #print(classid, class_list[classid], confidence, color)
    #cv2.imshow("output", frame)
    #cv2.waitKey()
    # Create a Named Window
    cv2.imwrite(" test1.jpg", frame)
    global path_to_img
    path_to_img = " test1.jpg"
    global is_new_image
    is_new_image = True
    '''
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        
    # Show the Image in the Window
    cv2.imshow("output", frame)
        
    # Resize the Window
    cv2.resizeWindow("output", 1920, 1080)
        
    # Wait for <> miliseconds
    cv2.waitKey()
    '''
    return class_list[classid], confidence

def main():
    '''
    Overview[0][0]="Object Detection"
    Overview[1][0]="Object Identified"
    Overview[2][0]="Object Located"
    Overview[3][0]="Object Grabbed"
    Overview[4][0]="Object Moving"
    Overview[5][0]="Object Dropped"
    Overview[6][0]="Cycle Complete"
    '''
    forceThread = Thread(target = poll_force2)
    forceThread.start()
    while True:
        #ALL RED
        time.sleep(1)
        #object detection YELLOW
        set(0, 1)
        distance = poll_distance_until()
        #object detection GREEN
        set(0, 2)
        set(1, 1)
        start = time.time()
        print("taking picture")
        take_picture()
        print("detecting picture...")
        #object identified YELLOW
        set(1, 1)
        object, confidence = detect_img(" test.jpg", start)
        #object identified GREEN
        set(1, 2)
        global Guess
        Guess = object
        global Confidence
        Confidence = confidence
        #open gripper, close gripper, open gripper
        #object located GREEN
        set(2, 2)
        time.sleep(5)
        #object grabbed YELLOW
        set(3, 1)
        if object=="bodywash":
            newopen, newclose = grip_bodywash()
        elif object=="deodorant":
            newopen, newclose = grip_deodorant()
        elif object=="soap":
            newopen, newclose = grip_soap()
        elif object=="spray":
            newopen, newclose = grip_spray()
        elif object=="tea":
            newopen, newclose = grip_tea()
        #object grabbed GREEN, object moving GREEN
        set(3, 2)
        set(4, 2)
        time.sleep(5)
        #open gripper
        #object dropped YELLOW
        set(5,1)
        open_steps(newopen)
        #object dropped GREEN, cycle complete YELLOW
        set(5,2)
        set(6,1)
        time.sleep(2)
        #close gripper
        reset_close(newclose)
        #cycle complete GREEN
        set(6,2)
        print("removing picture")
        remove_picture(" test.jpg")
        print("program end")
        reset()
