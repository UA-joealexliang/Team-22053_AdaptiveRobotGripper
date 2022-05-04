import threading, sys, os

theVar = 1

class MyThread ( threading.Thread ):

   def run ( self ):

      #global theVar
      #print('This is thread ' + str ( theVar ) + ' speaking.')
      print('Hello and good bye.')
      #theVar = theVar + 1
      #if theVar == 4:
          #sys.exit(1)
      os._exit(1)
      #print('(done)')

#for x in range( 7 ):
#   MyThread().start()

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
message = input("Press enter to quit\n\n") # Run until someone presses enter
GPIO.cleanup() # Clean up
