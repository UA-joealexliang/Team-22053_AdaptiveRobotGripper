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

poll_force(1, True)