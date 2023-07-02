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
import numpy as np
import pandas as pd
import HiwonderSDK.Sonar as Sonar
import HiwonderSDK.Board as Board
import HiwonderSDK.mecanum as mecanum
import HiwonderSDK.FourInfrared as FourInfrared
import imageio.v3 as iio
import matplotlib.pyplot as plt

##################################################
# SLAM Map
#
# This routine measures the distance to a feature wall
# running alongside the path of the robot. At the same
# time, the robot tracks its location by detecting the
# black tape stripes crossing the robot's path.
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
MAX_SPEED = 45.0 # mm/second - the maximum speed the robot can travel
MIN_DISTANCE_THRESHOLD = 5.0 # cm - if distance_mean to the wall is closer than this then stop the robot
MIN_SPEED = 35.0 # mm/second - the minimum speed the robot can travel if not stopping
MAX_LINE_COUNT = 8 # stop the robot after detecting this many lines across the track
BELIEVED_STARTING_POINT = -0.10 # The robot believes that it is starting 10 cm back from the first landmark
BELIEVED_DISTANCE_BETWEEN_WAYPOINTS = 0.02 # The robot will guess each measurement waypoint is x cm from the last
ACTUAL_DISTANCE_BETWEEN_LANDMARKS = 0.10 # The actual distance (in cm) between landmarks
LOW_CONFIDENCE_WAYPOINT = 1.0 # The low confidence of the guessed waypoint added to the omega_matrix and xi_vector
HIGH_CONFIDENCE_LANDMARK = 100.0 # The high confidence of the painted landmarks added to the omega_matrix and xi_vector

servo1_default_position = SERVO_DEFAULT_POSITION # These defaults will be updated by the fine tuning found in the YAML file
servo2_default_position = SERVO_DEFAULT_POSITION
distance_list = [] # List of distance measurements to the feature wall
landmark_list = [] # List of distance offsets when a landmark line is detected
line_latched = False # Latched True if any of the infrared sensors detect a line
line_detected = False # True for a single scan when a line is detected
line_count = 0 # Count the number of lines detected by the robot as it runs
speed = MAX_SPEED # The robot runs at a constant minimim speed

robot = mecanum.MecanumChassis()
sonar = Sonar.Sonar()
camera = Camera.Camera()
infrared = FourInfrared.FourInfrared()

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
    if DEBUG: print(f'[initialize] Board.setPWMServoPulse(2, {servo2_default_position}, 1000)')
    Board.setPWMServoPulse(2, servo2_default_position, 1000) # 500 = 0-degrees * 11.1 + 500 = point camera hard-right
    time.sleep(0.3)
    if DEBUG: print(f'[initialize] Board.setPWMServoPulse(2, {servo2_default_position}, 1000)')
    Board.setPWMServoPulse(2, servo2_default_position, 1000) # 500 = 0-degrees * 11.1 + 500 = point camera hard-right
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
    global line_count

    if DEBUG: print("[move]")
    while True:
        if is_running and line_count == 0:
            speed = MAX_SPEED
        elif is_running and line_count < MAX_LINE_COUNT:
            speed = MIN_SPEED
        else:
            is_running = False
            speed = 0.0
        if False and DEBUG: print(f'[move] robot.set_velocity({speed:.1f}, 90, 0)')
        robot.set_velocity(speed, 90, 0)

# Calculate the distance_mean from the robot to the wall
def run(camera_image):
    global distance_list
    global landmark_list
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

    # Collect measurements to the feature wall and offsets to landmarks
    if line_count > 0:
        if line_detected:
            landmark_list.append( len(distance_list) )
        if distance_sensor < MAX_DISTANCE_THRESHOLD:
            distance_list.append(distance_sensor)
        else:
            distance_list.append(np.nan)

    return cv2.putText(camera_image, f'Dist:{distance_sensor:.1f}cm Line:{line_count}', (30, 480-30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, IMAGE_TEXT_COLOR, 2)  # Update the camera image

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

def generate_omega_xi():
    """
    This routine converts the sample distance x-locations into the
    omega_matrix and xi_vector used by the Graph SLAM algorithm, as
    well as the y-distance measurements to the feature wall.
    """
    global distance_list
    global landmark_list

    omega_matrix = np.zeros([len(distance_list), len(distance_list)], dtype=float) # Initialize the datasets based upon the size of the actual measurements
    xi_vector = np.zeros(len(distance_list), dtype=float)
    for i in range(len(distance_list) - 1):
        # Add the naive distances between the waypoints to the omega_matrix and xi_vector
        omega_matrix[i, i] += 1 * LOW_CONFIDENCE_WAYPOINT # distance between waypoint i and waypoint i+1
        omega_matrix[i, i+1] += -1 * LOW_CONFIDENCE_WAYPOINT
        omega_matrix[i+1, i] += -1 * LOW_CONFIDENCE_WAYPOINT
        omega_matrix[i+1, i+1] += 1 * LOW_CONFIDENCE_WAYPOINT
        xi_vector[i] += -BELIEVED_DISTANCE_BETWEEN_WAYPOINTS * LOW_CONFIDENCE_WAYPOINT # location difference between waypoints
        xi_vector[i+1] += BELIEVED_DISTANCE_BETWEEN_WAYPOINTS * LOW_CONFIDENCE_WAYPOINT

        # Add the precise landmarks to the omega_matrix and xi_vector
        if i in landmark_list:
            landmark_location = landmark_list.index(i) * ACTUAL_DISTANCE_BETWEEN_LANDMARKS
            omega_matrix[i, i] += 1 * HIGH_CONFIDENCE_LANDMARK
            xi_vector[i] += landmark_location * HIGH_CONFIDENCE_LANDMARK

    return omega_matrix, xi_vector

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
            if GENERATE_MOVIE:
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

    # Run the SLAM algorithm to calculate the more accurate x_locations using the Graph SLAM algorithm to calculate mu
    omega_matrix, xi_vector = generate_omega_xi()
    calculated_mu_x = np.linalg.inv(np.matrix(omega_matrix)) * np.expand_dims(xi_vector, 0).transpose()
    if DEBUG: print(f'[main] line_count,{line_count},distance_list,{len(distance_list)},landmark_list,{len(landmark_list)},calculated_mu_x,{len(calculated_mu_x)},')

    # Plot the shape of the feature wall based upon the Graph SLAM calculations
    output_figure_filename = 'feature_wall.png'
    output_figure_filepath = IMAGES_DIRECTORY + '/' + output_figure_filename
    plt.title("Shape of the Feature Wall")
    plt.scatter(list(calculated_mu_x), list(distance_list), label='Feature Wall', s=5)
    plt.xlabel('x-location along 1-dimensional track')
    plt.ylabel('y-measurement to feature wall') 
    plt.savefig(output_figure_filepath)

    # Finish up and exit
    if DEBUG: print('[main] Done.')
