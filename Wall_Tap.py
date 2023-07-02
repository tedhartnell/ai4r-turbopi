#!/usr/bin/python3
#coding=utf8
import sys
sys.path.append('/home/pi/TurboPi/')
import cv2
import os
import shutil
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
import HiwonderSDK.FourInfrared as FourInfrared
import imageio.v3 as iio

##################################################
# PID Wall Tap
#
# This routine slows the robot down as it approaches a wall
# using a PID controller that will cause the robot to
# gently tap the wall before stopping.
#
# The robot will turn its camera to the right and take
# a sequence of photos as it moves. When it stops the
# robot will stitch the photos together into a gif movie.
#
# Along the way the robot will count the number of
# black lines it crosses over.
#
##################################################

# Check the version of Python running this code
if sys.version_info.major <= 2:
    print('Please run this program with python3!')
    sys.exit(0)

DEBUG = True
GENERATE_MOVIE = False
IMAGES_DIRECTORY = './images'
MOVIES_DIRECTORY = './movies'
IMAGE_TEXT_SIZE = 12
IMAGE_TEXT_COLOR = (0, 255, 255)

SERVO_DEFAULT_POSITION = 1500 # The default 90-degree position for both camera servos - may have been adjusted
SERVO_HARD_RIGHT_POSITION = 500 # Pulse Width = Angle-in-degrees * 11.1 + 500 - turn the camera servo hard-right
SERVO_HARD_LEFT_POSITION = 2500 # Turn the camera servo hard-left = 180 * 11.1 + 500
MAX_DISTANCE_THRESHOLD = 100.0 # cm - if greater distance_mean away from the wall than this run at full speed
MAX_SPEED = 100.0 # mm/second - the maximum speed the robot can travel
MIN_DISTANCE_THRESHOLD = 5.0 # cm - if distance_mean to the wall is closer than this then stop the robot
MIN_SPEED = 25.0 # mm/second - the minimum speed the robot can travel if not stopping
PID_PROPORTIONAL_COEFFICIENT = 2.0 # Multiply the distance_mean from the wall by this coefficient

servo1_default_position = SERVO_DEFAULT_POSITION # These defaults will be updated by the fine tuning found in the YAML file
servo2_default_position = SERVO_DEFAULT_POSITION
distance_mean = MAX_DISTANCE_THRESHOLD # cm - the mean distance to the wall
distance_last = MAX_DISTANCE_THRESHOLD # cm - distances can only decrease as the wall approaches
distance_list = []
line_latched = False # Latched True if any of the infrared sensors detect a line
line_detected = False # True for a single scan when a line is detected
line_count = 0 # Count the number of lines detected by the robot as it runs
speed = MAX_SPEED


robot = mecanum.MecanumChassis()
sonar = Sonar.Sonar()
camera = Camera.Camera()
infrared = FourInfrared.FourInfrared()

stop_motor = True
is_running = False

# Initialize the robot's hardware
def initialize():
    if DEBUG: print("[initialize]")

    # PNG / JPG images go to the images director
    # GIF is created from the plots and saved in the movies directory
    if not os.path.exists(IMAGES_DIRECTORY): os.makedirs(IMAGES_DIRECTORY)
    if not os.path.exists(MOVIES_DIRECTORY): os.makedirs(MOVIES_DIRECTORY)

    # Clear out the old images from the images directory
    folder = IMAGES_DIRECTORY
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as exception:
            print(f'Failed to delete the file {file_path}. Reason: {exception}')

    # Clear out the old gif files from the movies directory
    folder = MOVIES_DIRECTORY
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as exception:
            print(f'Failed to delete the file {file_path}. Reason: {exception}')

    # Stop the robot if it was moving
    if DEBUG: print(f'[initialize] robot.set_velocity(0, 90, 0)')
    robot.set_velocity(0, 90, 0) # velocity (mm/s), direction (0-360 with 90=straight-ahead), angular rate (speed at which chassis rotates between -2 and 2)

    # Get the default camera servo positions
    servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path) # Calibrated servo positions: {'servo1': 1440, 'servo2': 1440}
    servo1_default_position = servo_data['servo1']
    servo2_default_position = servo_data['servo2']

    # Prepare to move the camera into position to take a sequence of images
    if DEBUG: print(f'[initialize] Board.setPWMServoPulse(2, {SERVO_HARD_RIGHT_POSITION}, 1000)')
    Board.setPWMServoPulse(2, SERVO_HARD_RIGHT_POSITION, 1000) # 500 = 0-degrees * 11.1 + 500 = point camera hard-right
    time.sleep(0.3)
    if DEBUG: print(f'[initialize] Board.setPWMServoPulse(2, {SERVO_HARD_RIGHT_POSITION}, 1000)')
    Board.setPWMServoPulse(2, SERVO_HARD_RIGHT_POSITION, 1000) # 500 = 0-degrees * 11.1 + 500 = point camera hard-right
    time.sleep(1) # Move 1000 ms = 1 second

    # Turn the camera on
    if DEBUG: print('[initialize] camera.camera_open(correction=True)')
    camera.camera_open(correction=True) # Take an image from the camera
    time.sleep(1.0) # Allow time for the camera to turn on

    # Turn the sonar colors on
    if DEBUG: print(f'[initialize] sonar.setPixelColor(0, Board.PixelColor(255, 0, 0))')
    sonar.setPixelColor(0, Board.PixelColor(255, 0, 0))
    if DEBUG: print(f'[initialize] sonar.setPixelColor(1, Board.PixelColor(255, 0, 0))')
    sonar.setPixelColor(1, Board.PixelColor(255, 0, 0))

# Move the robot each iteration - this routine runs independently in a thread
def move():
    global is_running
    global distance_mean

    if DEBUG: print("[move]")
    while True:
        if is_running:
            if math.isnan(distance_mean):
                continue
            if distance_mean > MAX_DISTANCE_THRESHOLD:
                speed = MAX_SPEED
            elif distance_mean <= MIN_DISTANCE_THRESHOLD:
                forward = False
                stop_motor = True
                is_running = False
                speed = 0.0
            else:
                speed = min( max(distance_mean * PID_PROPORTIONAL_COEFFICIENT, MIN_SPEED), MAX_SPEED )
        else:
            speed = 0.0
        if False and DEBUG: print(f'[move] robot.set_velocity({speed:.1f}, 90, 0)')
        robot.set_velocity(speed, 90, 0)

# Calculate the distance_mean from the robot to the wall
def run(camera_image):
    global distance_mean
    global distance_last
    global distance_list
    global line_latched
    global line_detected
    global line_count

    # Get the raw distance measurement from the sensor
    distance_sensor = sonar.getDistance() / 10.0 # Maximum sensor accuracy = 40 cm
    if False and DEBUG: print(f'[run] distance_sensor={distance_sensor}')

    # Get the four infrared line detector sensors (each is True or False)
    # line_detected is set True for just a single scan
    line_list = infrared.readData()
    count_detected = 0
    for line in line_list:
        if line: count_detected += 1
    if not line_latched and count_detected > 0:
        line_detected = True
        line_latched = True
    elif count_detected == 0:
        line_detected = False
        line_latched = False
    else:
        line_detected = False
    if line_detected:
        line_count += 1
        if DEBUG: print(f'[run] line_count={line_count}')

    # Ignore bad sensor measurements if the wall is getting further away 
    if distance_sensor <= MAX_DISTANCE_THRESHOLD: # distance_last
        distance_last = distance_sensor
        distance_list.append(distance_sensor)
        dataframe = pd.DataFrame(distance_list)
        dataframe_ = dataframe.copy()
        mu = dataframe_.mean()
        std = dataframe_.std()

        data_clean = dataframe[np.abs(dataframe - mu) <= std]
        distance_mean = data_clean.mean()[0]

        if len(distance_list) >= 5:
            distance_list.remove(distance_list[0])

    return cv2.putText(camera_image, f'Dist:{distance_mean:.1f}cm Line:{line_count}', (30, 480-30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, IMAGE_TEXT_COLOR, 2)  # Update the camera image

# Stop the robot and restore defaults
def stop():
    if DEBUG: print("[stop]")
    is_running = False

    # Stop the robot from moving - stopping twice was part of the original code
    if DEBUG: print(f'[stop] robot.set_velocity(0, 90, 0)')
    robot.set_velocity(0, 90, 0)
    time.sleep(0.3)
    if DEBUG: print(f'[stop] robot.set_velocity(0, 90, 0)')
    robot.set_velocity(0, 90, 0)

    # Restore the camera to the default position
    if DEBUG: print(f'[stop] Board.setPWMServoPulse(2, {servo2_default_position}, 1000)')
    Board.setPWMServoPulse(2, servo2_default_position, 1000) # Camera Left/Right
    time.sleep(0.3)
    if DEBUG: print(f'[stop] Board.setPWMServoPulse(2, {servo2_default_position}, 1000)')
    Board.setPWMServoPulse(2, servo2_default_position, 1000) # Camera Left/Right
    time.sleep(1) # Move 1000 ms = 1 second

    # Turn off the camera
    camera.camera_close()

    # Switch off the sonar lights
    if DEBUG: print(f'[stop] sonar.setPixelColor(0, Board.PixelColor(0, 0, 0))')
    sonar.setPixelColor(0, Board.PixelColor(0, 0, 0))
    if DEBUG: print(f'[stop] sonar.setPixelColor(1, Board.PixelColor(0, 0, 0))')
    sonar.setPixelColor(1, Board.PixelColor(0, 0, 0))

# Main routine
if __name__ == '__main__':
    if DEBUG: print('[main]')

    # Announce the initialization of the robot
    if DEBUG: print(f'[main] Board.setBuzzer(1)')
    Board.setBuzzer(1) # Turn on the Buzzer
    time.sleep(0.1) # Leave the Buzzer on for a short blip
    if DEBUG: print(f'[main] Board.setBuzzer(0)')
    Board.setBuzzer(0) # Turn off the Buzzer

    # Initialize the robot
    initialize()
    signal.signal(signal.SIGINT, stop)

    # Announce the start of motion
    if DEBUG: print(f'[main] Board.setBuzzer(1)')
    Board.setBuzzer(1) # Turn on the Buzzer
    time.sleep(1.0) # Leave the Buzzer on for a long blip
    if DEBUG: print(f'[main] Board.setBuzzer(0)')
    Board.setBuzzer(0) # Turn off the Buzzer

    # Start the robot running
    stop_motor = True
    is_running = True

    # Instantiate a thread to keep the robot moving
    move_thread = threading.Thread(target=move)
    move_thread.setDaemon(True)
    move_thread.start()

    # Collect images while the robot is running
    image_id = 1
    image_filenames = []
    while is_running:
        camera_image = camera.frame
        if camera_image is not None:
            frame = camera_image.copy() # Copy the image from the camera
            Frame = run(frame)  
            frame_resize = cv2.resize(Frame, (320, 240)) # Resize the image down to 320*240
            cv2.imshow('frame', frame_resize)
            image_filename = os.path.join(IMAGES_DIRECTORY, f'{image_id:03d}.jpg')
            cv2.imwrite(image_filename, frame) # Save the image
            image_filenames.append(image_filename)
            image_id += 1
            key = cv2.waitKey(1)
            if key == 27:
                break
        else:
            time.sleep(0.01)

    # Announce the stopping of motion
    if DEBUG: print(f'[main] Board.setBuzzer(1)')
    Board.setBuzzer(1) # Turn on the Buzzer
    time.sleep(0.1) # Leave the Buzzer on for a long blip
    if DEBUG: print(f'[main] Board.setBuzzer(0)')
    Board.setBuzzer(0) # Turn off the Buzzer

    # Stop the robot and close the camera windows
    stop()
    cv2.destroyAllWindows()

    # Create Gif Animation - Heatmap Actions
    if DEBUG: print(f'[main] Writing: {MOVIES_DIRECTORY}/camera_movie.gif')
    if GENERATE_MOVIE:
        camera_images = []
        for filename in image_filenames:
            camera_images.append(iio.imread(filename))
        iio.imwrite(f'{MOVIES_DIRECTORY}/camera_movie.gif', camera_images, loop=0, duration=100) # duration(in ms): `fps=50` == `duration=20` (1000 * 1/50)

    # Announce the exit of main
    if DEBUG: print(f'[main] Board.setBuzzer(1)')
    Board.setBuzzer(1) # Turn on the Buzzer
    time.sleep(1.0) # Leave the Buzzer on for a long blip
    if DEBUG: print(f'[main] Board.setBuzzer(0)')
    Board.setBuzzer(0) # Turn off the Buzzer

    if DEBUG: print(f'[main] line_count={line_count}')
    if DEBUG: print('[main] Done.')
