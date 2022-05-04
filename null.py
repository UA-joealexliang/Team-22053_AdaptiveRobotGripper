from threading import Thread
stop = True

def returnBad():
    return None

x = returnBad
print("Done")

def stopThread():
    while True:
        print("True")
        global stop
        if stop==False:
            break

def stop():
    global stop
    stop = False

stopT = Thread(target=stopThread)
stopT.start()
stop() 