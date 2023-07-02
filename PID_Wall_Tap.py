#!/usr/bin/python3
#coding=utf8
import sys
sys.path.append('/home/pi/TurboPi/')
import cv2
import os
import time
import math
import signal
import Camera
import threading
import numpy as np
import yaml_handle
import pandas as pd
import HiwonderSDK.Sonar as Sonar
import HiwonderSDK.Board as Board
import HiwonderSDK.mecanum as mecanum

##################################################
# PID Wall Tap
#
# This routine slows the robot down as it approaches a wall
# using a PID controller that will cause the robot to
# gently tap the wall before stopping.
#
##################################################

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)

car = mecanum.MecanumChassis()

DEBUG = True
MAX_DISTANCE_THRESHOLD = 100.0 # cm - if greater distance away from the wall than this run at full speed
MAX_SPEED = 100.0 # mm/second - the maximum speed the robot can travel
MIN_DISTANCE_THRESHOLD = 5.0 # cm - if distance to the wall is closer than this then stop the robot
MIN_SPEED = 25.0 # mm/second - the minimum speed the robot can travel if not stopping
distance = MAX_DISTANCE_THRESHOLD
distance_last = MAX_DISTANCE_THRESHOLD # cm - distances can only decrease as the wall approaches
distance_list = []
IMAGE_DIRECTORY = './images'
speed = MAX_SPEED

PID_PROPORTIONAL_COEFFICIENT = 2.0 # Multiply the distance from the wall by this coefficient

HWSONAR = None
stopMotor = True
__isRunning = False
servo1_default_position = 1500
servo2_default_position = 1500

TextSize = 12
TextColor = (0, 255, 255)

# Initialize the robot's hardware
def initMove():
    global DEBUG
    global servo1_default_position
    global servo2_default_position

    if DEBUG: print("[initMove]")
    if DEBUG: print(f'[initMove] car.set_velocity(0,90,0)')
    car.set_velocity(0,90,0) # velocity (mm/s), direction (0-360 with 90=straight-ahead), angular rate (speed at which chassis rotates between -2 and 2)
    servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path) # Calibrated servo positions: {'servo1': 1440, 'servo2': 1440}
    servo1_default_position = servo_data['servo1']
    servo2_default_position = servo_data['servo2']
#    if DEBUG: print(f'[initMove] Board.setPWMServoPulse(1, {servo1_default_position}, 1000)')
#    Board.setPWMServoPulse(1, servo1_default_position, 1000) # Camera Up/Down
#    time.sleep(1) # Move 1000 ms = 1 second
#    if DEBUG: print(f'[initMove] Board.setPWMServoPulse(2, {servo2_default_position}, 1000)')
#    Board.setPWMServoPulse(2, servo2_default_position, 1000) # Camera Left/Right
#    time.sleep(1) # Move 1000 ms = 1 second

    if DEBUG: print(f'[initMove] Board.setBuzzer(1)')
    Board.setBuzzer(1) # Turn on the Buzzer
    time.sleep(0.1) # Leave the Buzzer on for a short blip
    if DEBUG: print(f'[initMove] Board.setBuzzer(0)')
    Board.setBuzzer(0) # Turn off the Buzzer

    # Pulse Width = Angle-in-degrees * 11.1 + 500
    if DEBUG: print(f'[initMove] Board.setPWMServoPulse(2, 500, 1000)')
    Board.setPWMServoPulse(2, 500, 1000) # 500 = 0-degrees * 11.1 + 500 = point camera hard-right
    time.sleep(1) # Move 1000 ms = 1 second
    # Board.setPWMServoPulse(2, 1500, 1000) # 1500 = 90-degrees * 11.1 + 500 = point camera straight ahead
    # Board.setPWMServoPulse(2, 2500, 1000) # 2500 = 180-degrees * 11.1 + 500 = point camera hard-left

    if DEBUG: print(f'[initMove] Board.setBuzzer(1)')
    Board.setBuzzer(1) # Turn on the Buzzer
    time.sleep(1.0) # Leave the Buzzer on for a long blip
    if DEBUG: print(f'[initMove] Board.setBuzzer(0)')
    Board.setBuzzer(0) # Turn off the Buzzer

# Reset the robot's runtime parameters
def reset():
    global DEBUG
    global MAX_DISTANCE_THRESHOLD # cm - if greater distance away from the wall than this run at full speed
    global MAX_SPEED # mm/second - the maximum speed the robot can travel
    global speed
    global distance
    global stopMotor
    global __isRunning
    
    if DEBUG: print("[reset]")
    distance = MAX_DISTANCE_THRESHOLD
    speed = MAX_SPEED
    stopMotor = True
    __isRunning = False
    
# App Initialization
def init():
    global DEBUG
    if DEBUG: print("[init]")
    initMove()
    reset()
    
# App Start
__isRunning = False
def start():
    global DEBUG
    global __isRunning
    global stopMotor
    
    if DEBUG: print("[start]")
    stopMotor = True
    __isRunning = True

# App Stop
def stop():
    global DEBUG
    global servo1_default_position
    global servo2_default_position
    global __isRunning
    
    if DEBUG: print("[stop]")
    Board.setBuzzer(1) # Turn on the Buzzer
    time.sleep(0.1) # Leave the Buzzer on for a short blip
    if DEBUG: print(f'[stop] Board.setBuzzer(0)')
    Board.setBuzzer(0) # Turn off the Buzzer
    
    __isRunning = False
    if DEBUG: print(f'[stop] car.set_velocity(0,90,0)')
    car.set_velocity(0,90,0)
    time.sleep(0.3)
    if DEBUG: print(f'[stop] car.set_velocity(0,90,0)')
    car.set_velocity(0,90,0)
#    if DEBUG: print(f'[stop] Board.setPWMServoPulse(1, {servo1_default_position}, 1000)')
#    Board.setPWMServoPulse(1, servo1_default_position, 1000) # Camera Up/Down
#    time.sleep(1) # Move 1000 ms = 1 second
    if DEBUG: print(f'[stop] Board.setPWMServoPulse(2, {servo2_default_position}, 1000)')
    Board.setPWMServoPulse(2, servo2_default_position, 1000) # Camera Left/Right
    time.sleep(1) # Move 1000 ms = 1 second

    if DEBUG: print(f'[stop] Board.setBuzzer(1)')
    Board.setBuzzer(1) # Turn on the Buzzer
    time.sleep(1.0) # Leave the Buzzer on for a long blip
    if DEBUG: print(f'[stop] Board.setBuzzer(0)')
    Board.setBuzzer(0) # Turn off the Buzzer

# App Exit
def exit():
    global DEBUG
    global __isRunning
    __isRunning = False

    if DEBUG: print("[exit]")
    if DEBUG: print(f'[exit] car.set_velocity(0,90,0)')
    car.set_velocity(0,90,0)
    time.sleep(0.3)
    if DEBUG: print(f'[exit] car.set_velocity(0,90,0)')
    car.set_velocity(0,90,0)
    if DEBUG: print(f'[exit] HWSONAR.setPixelColor(0, Board.PixelColor(0, 0, 0))')
    HWSONAR.setPixelColor(0, Board.PixelColor(0, 0, 0))
    if DEBUG: print(f'[exit] HWSONAR.setPixelColor(1, Board.PixelColor(0, 0, 0))')
    HWSONAR.setPixelColor(1, Board.PixelColor(0, 0, 0))
    print("[exit] PID Wall Tap Exit")

# Set Speed from App
def setSpeed(args):
    global DEBUG
    global speed

    if DEBUG: print("[exit]")
    speed = int(args[0])
    return (True, ())

# Move the robot each iteration
def move():
    global DEBUG
    global MAX_DISTANCE_THRESHOLD # cm - if greater distance away from the wall than this run at full speed
    global MAX_SPEED # mm / second - the maximum speed the robot can travel
    global MIN_DISTANCE_THRESHOLD # cm - if distance to the wall is closer than this then stop the robot
    global MIN_SPEED # mm/second - the minimum speed the robot can travel if not stopping
    global PID_PROPORTIONAL_COEFFICIENT # Multiply the distance from the wall by this coefficient

    global speed
    global distance
    global stopMotor
    global __isRunning

    if DEBUG: print("[move]")
    while True:
        if __isRunning:
            if math.isnan(distance):
                continue
            if distance > MAX_DISTANCE_THRESHOLD:
                speed = MAX_SPEED
            elif distance <= MIN_DISTANCE_THRESHOLD:
                forward = False
                stopMotor = True
                __isRunning = False
                speed = 0.0
            else:
                speed = min( max(distance * PID_PROPORTIONAL_COEFFICIENT, MIN_SPEED), MAX_SPEED )
        else:
            speed = 0.0
        if False and DEBUG: print(f'[move] car.set_velocity({speed:.1f}, 90, 0)')
        car.set_velocity(speed, 90, 0)
 
# Instantiate a thread to keep the robot moving
th = threading.Thread(target=move)
th.setDaemon(True)
th.start()

# Calculate the distance from the robot to the wall
def run(img):
    global DEBUG
    global HWSONAR
    global distance
    global distance_list
    global distance_last # cm - distances can only decrease as the wall approaches
    
    distance_sensor = HWSONAR.getDistance() / 10.0 # Maximum sensor accuracy = 40 cm
    if False and DEBUG: print(f'[run] distance_sensor={distance_sensor}')

    # Ignore bad sensor measurements if the wall is getting further away 
    if distance_sensor <= distance_last:
        distance_last = distance_sensor
        distance_list.append(distance_sensor)
        dataframe = pd.DataFrame(distance_list)
        dataframe_ = dataframe.copy()
        mu = dataframe_.mean()
        std = dataframe_.std()

        data_clean = dataframe[np.abs(dataframe - mu) <= std]
        distance = data_clean.mean()[0]

        if len(distance_list) >= 5:
            distance_list.remove(distance_list[0])

    return cv2.putText(img, "Dist:%.1fcm"%distance, (30, 480-30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, TextColor, 2)  # Update the camera image


# Manually stop the robot from the App
def manual_stop(signum, frame):
    global DEBUG
    global __isRunning
    
    print('[manual_stop]')
    __isRunning = False
    if DEBUG: print(f'[manual_stop] car.set_velocity(0,90,0)')
    car.set_velocity(0,90,0)

if __name__ == '__main__':
    if DEBUG: print('[main]')
    init()
    HWSONAR = Sonar.Sonar()
    camera = Camera.Camera()
    if DEBUG: print('[main] camera.camera_open(correction=True)')
    camera.camera_open(correction=True) # Take an image from the camera
    time.sleep(1.0) # Allow time for the camera to turn on
    signal.signal(signal.SIGINT, manual_stop)
    start()
    image_id = 1
    while __isRunning:
        img = camera.frame
        if img is not None:
            frame = img.copy() # Copy the image from the camera
            Frame = run(frame)  
            frame_resize = cv2.resize(Frame, (320, 240)) # Resize the image down to 320*240
            cv2.imshow('frame', frame_resize)
            cv2.imwrite(os.path.join(IMAGE_DIRECTORY, f'{image_id:03d}.jpg'), frame)
            image_id += 1
            key = cv2.waitKey(1)
            if key == 27:
                break
        else:
            time.sleep(0.01)
    camera.camera_close()
    cv2.destroyAllWindows()
    stop()
    if DEBUG: print('[main] Done.')
