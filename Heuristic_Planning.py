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
# Heuristic Planning
#
# This routine finds the optimal path from the start
# to the goal using heuristics. If the robot finds
# a hole on the journey, then the robot will
# recalculate the plan and continue.
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
SERVO_HARD_UP_POSITION = 500 # Turn the camera servo hard-up = 180 * 11.1 + 500
SERVO_HARD_DOWN_POSITION = 2500 # Pulse Width = Angle-in-degrees * 11.1 + 500 - turn the camera servo hard-down
MAX_DISTANCE_THRESHOLD = 100.0 # cm - if greater distance_mean away from the wall than this run at full speed
MAX_SPEED = 100.0 # mm/second - the maximum speed the robot can travel
MIN_DISTANCE_THRESHOLD = 5.0 # cm - if distance_mean to the wall is closer than this then stop the robot
MIN_SPEED = 50.0 # mm/second - the minimum speed the robot can travel if not stopping
MAX_LINE_COUNT = 8 # stop the robot after detecting this many lines across the track
FORWARD_DISTANCE_PER_ITERATION = 54.0 / 180 # cm (measured at speed = 50) - the distance traveled by the robot per iteration when traveling forward
RIGHT_DISTANCE_PER_ITERATION = 54.0 / 800 # cm - the distance traveled by the robot per iteration when traveling right
FORWARD_DISTANCE_PER_STATE = 27.0 # cm - the distance traveled forward between states
RIGHT_DISTANCE_PER_STATE = 27.0 # cm - the distance traveled right between states

# Robot Hardware
robot = mecanum.MecanumChassis()
sonar = Sonar.Sonar()
camera = Camera.Camera()
infrared = FourInfrared.FourInfrared()

# Robot State Variables
servo1_default_position = SERVO_DEFAULT_POSITION # These defaults will be updated by the fine tuning found in the YAML file
servo2_default_position = SERVO_DEFAULT_POSITION
line_latched = False # Latched True if any of the infrared sensors detect a line
line_detected = False # True for a single scan when a line is detected
line_count = 0 # Count the number of lines detected by the robot as it runs
speed = MIN_SPEED # The robot runs at a constant minimim speed

# Robot Runtime Variables
is_running = False
forward_distance = 0.0 # The cumulative distance traveled in the forward-direction 
right_distance = 0.0 # The cumulative distance traveled in the right-direction

# Compass and Navigation
compass_not_applicable = -1
compass_north = 0
compass_west = 1
compass_south = 2
compass_east = 3

# Grid World Format:
#   0 = Navigable State
#   1 = Occupied State
grid_world   = [[0, 0, 0], # row x column
                [0, 1, 0],
                [0, 0, 0]]
state_grid   = [[]] # The policy grid filled in by the routine generate_comprehensive_policy_grid()

# Key grid locations
start_location = [2, 0, compass_east] # [row, column, direction] # Direction = N,W,S,E
goal_location = [0, 2, compass_west] # [row, column, direction]
current_location = start_location

# Compass Actions
compass_actions =  [[-1,  0], # Go North
                    [ 0, -1], # Go West
                    [ 1,  0], # Go South
                    [ 0,  1]] # Go East
compass_names = ['^', '<', 'v', '>', '*']
compass_angles = [90, 180, 270, 0, 90] # Direction angles used by the robot

# Turn Actions
turn_actions = [0, 1, 2, 3] # Go Straight / Turn Left / U-Turn / Turn Right
turn_names = ['#', 'L', 'U', 'R']
turn_costs = [1, 15, 37, 2] # Turning Left is more costly than Turning Right (U-Turn is the worst)

# State Class
state_class_id_counter = 0 # Automatically increment the state ID counter whenever a new state is created
class state_class:
    def __init__(self, row, column, direction):
        if row < 0 or row >= len(grid_world):
            raise ValueError("Row coordinate out of bounds")
        if column < 0 or column >= len(grid_world[0]):
            raise ValueError("Column coordinate out of bounds")
        global state_class_id_counter
        self.id = state_class_id_counter
        state_class_id_counter += 1
        self.row = row
        self.column = column
        self.direction = direction # Direction = N,W,S,E

    def __str__(self):
        if self.best_compass_action_id >= 0:
            return f'[row={self.row}, column={self.column}, direction={self.direction}{compass_names[self.direction]}, turn_action={self.best_turn_action_id}{turn_names[self.best_turn_action_id]}, goal_value={self.goal_value:.3f}]'
        return f'[row={self.row}, column={self.column}, direction={self.direction}, goal_value={self.goal_value:.3f}]'

    def is_goal(self):
        if self.row == goal_location[0] and self.column == goal_location[1]:
            if self.direction == goal_location[2] or goal_location[2] == compass_not_applicable:
                return True
        return False

def pretty_print_2d(array, as_integers=False):
    for i in range(len(array)):
        if i > 0: print()
        for j in range(len(array[i])):
            if as_integers:
                print(f'{array[i][j]:.0f}  ', end='')
            else:
                print(f'{array[i][j]:.3f}  ', end='')
    print()

def pretty_print_2d_characters(array, characters):
    for i in range(len(array)):
        if i > 0: print()
        for j in range(len(array[i])):
            if array[i][j] >= 0:
                print(f'{characters[array[i][j]]}  ', end='')
            else:
                print(f'.  ', end='')
    print()

def generate_heuristic_world(check_grid_world=False):
    # Heuristics are not yet intelligent enough to understand how to change direction
    maximum_possible_path_length = len(grid_world) * len(grid_world[0]) * max(turn_costs)
    state_grid = [[maximum_possible_path_length for row in range(len(grid_world[0]))] for column in range(len(grid_world))]
    state_grid[goal_location[0]][goal_location[1]] = 0
    found_update = True
    while found_update:
        found_update = False
        for row in range(len(grid_world)):
            for column in range(len(grid_world[row])):
                if check_grid_world:
                    if grid_world[row][column] > 0: # Check that the state is not a barrier
                        continue
                for compass_action_id in range(len(compass_actions)):
                    action = compass_actions[compass_action_id]
                    adjacent_row = row + action[0]
                    adjacent_column = column + action[1]
                    if adjacent_row < 0 or adjacent_row >= len(grid_world):
                        continue
                    if adjacent_column < 0 or adjacent_column >= len(grid_world[0]):
                        continue
                    # Heuristic optimistically assumes lowest turning cost
                    if state_grid[row][column] > state_grid[adjacent_row][adjacent_column] + min(turn_costs):
                        state_grid[row][column] = state_grid[adjacent_row][adjacent_column] + min(turn_costs)
                        found_update = True
    return state_grid

def generate_comprehensive_policy_grid(heuristic_world):
    # The grid containing all policy actions from every state
    state_grid = [[-1 for row in range(len(grid_world[0]))] for column in range(len(grid_world))]
    for row in range(len(grid_world)):
        for column in range(len(grid_world[row])):
            optimal_compass_action_id = -1
            optimal_action_value = heuristic_world[row][column]
            for compass_action_id in range(len(compass_actions)):
                compass_action = compass_actions[compass_action_id]
                adjacent_row = row + compass_action[0]
                adjacent_column = column + compass_action[1]
                if adjacent_row < 0 or adjacent_row >= len(grid_world):
                    continue
                if adjacent_column < 0 or adjacent_column >= len(grid_world[0]):
                    continue
                if grid_world[row][column] > 0: # Check that the from-state is not a barrier
                    continue
                if grid_world[adjacent_row][adjacent_column] > 0: # Check that the to-state is not a barrier
                    continue
                if heuristic_world[adjacent_row][adjacent_column] < optimal_action_value:
                    optimal_compass_action_id = compass_action_id
                    optimal_action_value = heuristic_world[adjacent_row][adjacent_column]
            if optimal_compass_action_id == -1 and grid_world[row][column] == 0:
                state_grid[row][column] = len(compass_names)-1 # '*' goal or local sink
            else:
                state_grid[row][column] = optimal_compass_action_id # best action
    return state_grid

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

    # [Up/Down] Prepare to move the camera into position to take a sequence of images
    if DEBUG: print(f'[initialize] Board.setPWMServoPulse(1, {SERVO_HARD_DOWN_POSITION}, 1000)')
    Board.setPWMServoPulse(1, SERVO_HARD_DOWN_POSITION, 1000) # 500 = 0-degrees * 11.1 + 500 = point camera hard-right
    time.sleep(0.3)
    if DEBUG: print(f'[initialize] Board.setPWMServoPulse(1, {SERVO_HARD_DOWN_POSITION}, 1000)')
    Board.setPWMServoPulse(1, SERVO_HARD_DOWN_POSITION, 1000) # 500 = 0-degrees * 11.1 + 500 = point camera hard-right
    time.sleep(1) # Move 1000 ms = 1 second

    # [Left/Right] Prepare to move the camera into position to take a sequence of images
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

    # Turn the sonar colors on - Green
    if DEBUG: print(f'[initialize] sonar.setPixelColor(0, Board.PixelColor(0, 255, 0))')
    sonar.setPixelColor(0, Board.PixelColor(0, 255, 0))
    if DEBUG: print(f'[initialize] sonar.setPixelColor(1, Board.PixelColor(0, 255, 0))')
    sonar.setPixelColor(1, Board.PixelColor(0, 255, 0))

# Move the robot each iteration - this routine runs independently in a thread
def move():
    global is_running
    global line_count
    global state_grid
    global goal_location
    global current_location
    global compass_angles
    global forward_distance
    global right_distance
    if DEBUG: print("[move]")

    # Move the robot according to the plan
    iteration = 0
    previous_location = None
    while True:
        if is_running and line_count == 0:
            
            # Check if the robot has reached the goal
            if current_location[0] == goal_location[0] and current_location[1] == goal_location[1]:
                is_running = False
                speed = 0.0
                continue

            # Check if the robot has hit a barrier
            if state_grid[current_location[0]][current_location[1]] < 0:
                is_running = False
                speed = 0.0
                continue

            # Set the robot speed and direction using the policy
            speed = MIN_SPEED
            current_location[2] = state_grid[current_location[0]][current_location[1]]

            # Estimate the absolute location (in cm) of the robot
            if previous_location != None:
                # Aggregate the direction traveled
                if previous_location[2] == compass_north:
                    forward_distance += FORWARD_DISTANCE_PER_ITERATION
                if previous_location[2] == compass_west:
                    right_distance -= RIGHT_DISTANCE_PER_ITERATION
                if previous_location[2] == compass_south:
                    forward_distance -= FORWARD_DISTANCE_PER_ITERATION
                if previous_location[2] == compass_east:
                    right_distance += RIGHT_DISTANCE_PER_ITERATION
            previous_location = current_location

            # Estimate the state location in the grid_world of the robot
            current_location[0] = forward_distance // FORWARD_DISTANCE_PER_STATE
            if current_location[0] < 0: current_location[0] = 0
            if current_location[0] > len(grid_world) - 1: current_location[0] = len(grid_world) - 1
            current_location[1] = right_distance // RIGHT_DISTANCE_PER_STATE
            if current_location[1] < 0: current_location[1] = 0
            if current_location[1] > len(grid_world[0]) - 1: current_location[1] = len(grid_world[0]) - 1
            iteration += 1
        else:
            is_running = False # The robot should stop immediately if it hits a out-of-bounds black line
            speed = 0.0
        
        # Set the robot's speed and direction
        direction_angle = compass_angles[ current_location[2] ]
        if DEBUG and is_running: print(f'[move] iteration,{iteration},forward_distance,{forward_distance},right_distance,{right_distance},current_location,{current_location}: robot.set_velocity({speed:.1f}, {direction_angle}, 0)')
        robot.set_velocity(speed, direction_angle, 0)

# Collect the runtime sensor data from the robot hardware
def run(camera_image):
    global is_running
    global distance_list
    global landmark_list
    global line_latched
    global line_detected
    global line_count
    global current_location
    if False and DEBUG: print(f'[run]')

    # Get the raw distance measurement from the sensor
    distance_sensor = sonar.getDistance() / 10.0 # Maximum sensor accuracy = 40 cm
    if False and DEBUG: print(f'[run] distance_sensor={distance_sensor}')

    # Get the four infrared line detector sensors (each is True or False)
    # line_detected is set True for just a single scan
    line_list = infrared.readData()
    infrared_sensors_detected = 0
    for line in line_list:
        if line: infrared_sensors_detected += 1
    if not line_latched and infrared_sensors_detected > 0:
        line_detected = True
        line_latched = True
    elif infrared_sensors_detected == 0:
        line_detected = False
        line_latched = False
    else:
        line_detected = False
    if line_detected:
        line_count += 1
        if DEBUG: print(f'[run] line_count={line_count}')

    return cv2.putText(camera_image, f'Location:({current_location[0]},{current_location[1]}) Distance:{distance_sensor:.1f}cm', (30, 480-30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, IMAGE_TEXT_COLOR, 2)  # Update the camera image

# Stop the robot and restore defaults
def stop():
    global is_running
    global servo1_default_position
    global servo2_default_position
    if DEBUG: print("[stop]")

    # Stop the robot from moving - stopping twice was part of the original code - also added motor hard-stops
    is_running = False
    if DEBUG: print(f'[stop] robot.set_velocity(0, 90, 0)')
    robot.set_velocity(0, 90, 0)
    time.sleep(0.3)
    if DEBUG: print(f'[stop] robot.set_velocity(0, 90, 0)')
    robot.set_velocity(0, 90, 0)
    time.sleep(0.3)
    if DEBUG: print(f'[stop] Board.setMotor(1, 0)')
    Board.setMotor(1, 0)
    if DEBUG: print(f'[stop] Board.setMotor(2, 0)')
    Board.setMotor(2, 0)
    if DEBUG: print(f'[stop] Board.setMotor(3, 0)')
    Board.setMotor(3, 0)
    if DEBUG: print(f'[stop] Board.setMotor(4, 0)')
    Board.setMotor(4, 0)

    # [Up/Down] Restore the camera to the default position
    if DEBUG: print(f'[stop] Board.setPWMServoPulse(1, {servo1_default_position}, 1000)')
    Board.setPWMServoPulse(1, servo1_default_position, 1000) # Camera Left/Right
    time.sleep(0.3)
    if DEBUG: print(f'[stop] Board.setPWMServoPulse(1, {servo1_default_position}, 1000)')
    Board.setPWMServoPulse(1, servo1_default_position, 1000) # Camera Left/Right
    time.sleep(1) # Move 1000 ms = 1 second

    # [Left/Right] Restore the camera to the default position
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
def main():
    global is_running
    global state_grid
    if DEBUG: print('[main]')

    # Calculate an heuristic that will quickly move the robot towards the goal
    print('Heuristic World:')
    heuristic_world = generate_heuristic_world(check_grid_world=True)
    pretty_print_2d(heuristic_world, True)

    # Generate the motion policy
    print('Comprehensive Policy Action Grid:')
    state_grid = generate_comprehensive_policy_grid(heuristic_world)
    pretty_print_2d_characters(state_grid, compass_names)

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

    # Finish up and exit
    if DEBUG: print('[main] Done.')

if __name__ == '__main__':
    main()
