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
path_to_img = 'Capture2.JPG'
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
def add(n):
    Overview[n][1] += 1
    #if Overview[n][1] == 1:
    #    P_v[n].config(bg="yellow")
    if Overview[n][1] == 1:
        P_v[n].config(bg="green")
    else:
        Overview[n][1] = 0
        P_v[n].config(bg="red")

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
        set(i, 0)
    for i in range(0,7):
        print(Overview[i][1])
    #for i in range(0,7):
    #    check()

'''
#Button to increase value
btn_increase0 = ttk.Button(master=window, text="+", command = lambda: add(0))
btn_increase1 = ttk.Button(master=window, text="+", command = lambda: add(1))
btn_increase2 = ttk.Button(master=window, text="+", command = lambda: add(2))
btn_increase3 = ttk.Button(master=window, text="+", command = lambda: add(3))
btn_increase4 = ttk.Button(master=window, text="+", command = lambda: add(4))
btn_increase5 = ttk.Button(master=window, text="+", command = lambda: add(5))
btn_increase6 = ttk.Button(master=window, text="+", command = lambda: add(6))
btn_increase7 = ttk.Button(master=window, text="+", command = lambda: add(7))
btn_increase8 = ttk.Button(master=window, text="+", command = lambda: add(8))
btn_increase9 = ttk.Button(master=window, text="+", command = lambda: add(9))
#Lays out all 7 buttons
btn_increase0.grid(row=0,column=2, sticky="w")
btn_increase1.grid(row=1,column=2, sticky="w")
btn_increase2.grid(row=2,column=2, sticky="w")
btn_increase3.grid(row=3,column=2, sticky="w")
btn_increase4.grid(row=4,column=2, sticky="w")
btn_increase5.grid(row=5,column=2, sticky="w")
btn_increase6.grid(row=6,column=2, sticky="w")
'''
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

"""
To do:
Calulate and display pick rate
Integrate
Set up cycle for all steps and auto reset
"""

#window.mainloop()
#cannot access is_new_image under a function
def guiThread():
    global is_new_image
    print(is_new_image)
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

def Test_Thread():
    sleep(3)
    global path_to_img
    path_to_img = "Capture3.JPG"
    global is_new_image
    is_new_image = True
    set(0, 1)
    global Guess
    Guess = "bodywash"
    global Confidence
    Confidence = 0.957
    sleep(4)
    reset()

'''
def GUI_Thread():
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
'''

            
newThread = Thread(target = Test_Thread)
newThread.start()
newThread2 = Thread(target=guiThread)
newThread2.start()
guiThread()
