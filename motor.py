from time import sleep
import RPi.GPIO as GPIO

DIR = 20 #Direction GPIO Pin
STEP = 21 #Step GPIO PIN
CW = 1 #CLOCKWISE ROTATION
CCW = 0 #CCW ROTATION
SPR = 800 #steps per revolution

GPIO.setmode(GPIO.BCM)
GPIO.setup(DIR,GPIO.OUT)
GPIO.setup(STEP,GPIO.OUT)
GPIO.output(DIR,CW)

MODE = (14,15,18) #MICROSTEP RESOLUTION GPIO
GPIO.setup(MODE, GPIO.OUT)
RESOLUTION = {'Full':(0,0,0),
              'Half':(1,0,0),
              '1/4':(0,1,0),
              '1/8':(1,1,0),
              '1/16':(0,0,1),
              '1/32':(1,0,1)}
GPIO.output(MODE,RESOLUTION['1/32'])

step_count = 200 * 32 #SPR
delay = .0005 / 32

force = 0
stepcnt = 0
open = 0
close = 0
refresh = .0001
object = 1
Ftest = 5
force_target = 0 # req'd force to grab
force_max = 0 #maximum force allowed
error_allowed = 0 #allowable error



def open_steps(steps):
    GPIO.output(DIR,CW)
    for x in range(steps):
            GPIO.output(STEP, GPIO.HIGH)
            sleep(delay)
            GPIO.output(STEP, GPIO.LOW)
            sleep(delay)
           
def close_steps(steps):
    GPIO.output(DIR,CCW)
    for x in range(steps):
        GPIO.output(STEP, GPIO.HIGH)
        sleep(delay)
        GPIO.output(STEP, GPIO.LOW)
        sleep(delay)

while 1:
    while force == 0:
        #close Z steps
        close_steps(Z)
        #update force applied
        force = force_input
       
    # open predetermined number of steps, Q
    open_steps(Q)
    #update step count
    stepcnt = Q
   
   
   
    #wait for trigger to grab object
    while !close:
        sleep(refresh)
   
    if object == 1: # shampoo
        force_target = A # req'd force to grab shampoo
        force_max = D #maximum force allowed
        error_allowed = B #allowable error
        close_steps(C) #close predetermined number of steps, C for shampoo
        stepcnt = stepcnt-C
       
    else if object == 2: #deodorant
        force_target = A # req'd force to grab
        force_max = D #maximum force allowed
        error_allowed = B #allowable error
        close_steps(C) #close predetermined number of steps, C
        stepcnt = stepcnt-C
       
    else if object == 3: #Soap
        force_target = A # req'd force to grab
        force_max = D #maximum force allowed
        error_allowed = B #allowable error
        close_steps(C) #close predetermined number of steps, C
        stepcnt = stepcnt-C
   
    else if object == 4: #spray
        force_target = A # req'd force to grab
        force_max = D #maximum force allowed
        error_allowed = B #allowable error
        close_steps(C) #close predetermined number of steps, C
        stepcnt = stepcnt-C
       
    else if object == 5: #tea
        force_target = A # req'd force to grab
        force_max = D #maximum force allowed
        error_allowed = B #allowable error
        close_steps(C) #close predetermined number of steps, C
        stepcnt = stepcnt-C
       
    error = abs(force_target - force_input)
   
    while (error > error_allowed) and (force_input < force_max):
       
        close_steps(Ftest) #close Ftest steps
        stepcnt = stepcnt - Ftest
        error = abs(force_target - force_input)
       
    #perhaps add additional tightening if above is insufficient
   
    #wait for trigger to open release object
    while !open:
        sleep(refresh)
       
    open_steps(Q-stepcnt)