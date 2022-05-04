import cv2
import time
import sys
import numpy as np
import subprocess
import os
import time

#import libraries for distance sensor
import RPi.GPIO as GPIO 
import time

#import libraries for pressure sensor
import spidev
import time

#Distance Sensor Constants
GET_DISTANCE_DELAY = 0.01
PRINT_DISTANCE = False
DISTANCE_SENSOR_LOADED =  False

#Force Sensor Constants
GET_FORCE_DELAY = 0.01
PRINT_FORCE = False
pad_channel = 0
measured_weight = 10
measured_bit = 1023
measured_n = measured_bit/(measured_weight*9.807) #max value is 1023 bits, corresponding to 10kg
#Create SPI
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz=1000000

#https://thepihut.com/blogs/raspberry-pi-tutorials/hc-sr04-ultrasonic-range-sensor-on-the-raspberry-pi
def distance_sensor():
    #first time? set everything up
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

def poll_distance(delay=GET_DISTANCE_DELAY, print=GET_DISTANCE_DELAY):
    try:
        while True:
            distance = distance_sensor()
            if print==True:
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

def poll_force(delay=GET_FORCE_DELAY, print_force=PRINT_FORCE):
    try:
        while True:
            force, pad_value = force_sensor()
            if print_force==True:
                print("Pressure Pad Value: %d" % pad_value)
                #print("Pressure Pad Force: %f" % force)
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

#constants
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
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

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
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        
    # Show the Image in the Window
    cv2.imshow("output", frame)
        
    # Resize the Window
    cv2.resizeWindow("output", 1920, 1080)
        
    # Wait for <> miliseconds
    cv2.waitKey()

def main():
    start = time.time()
    print("taking picture")
    take_picture()
    print("detecting picture...")
    detect_img(" test.jpg", start)
    print("removing picture")
    remove_picture(" test.jpg")
    print("program end")

#main()
poll_distance()