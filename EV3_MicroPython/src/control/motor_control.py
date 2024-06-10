
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor
from pybricks.robotics import DriveBase
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.media.ev3dev import SoundFile, ImageFile
from vision import camera_control

import cv2 
import numpy as np



def control():

    
    
    ev3 = EV3Brick()
    


    left_motor = Motor(Port.B,Direction.COUNTERCLOCKWISE)
    right_motor = Motor(Port.C,Direction.COUNTERCLOCKWISE)
    gripper_motor = Motor(Port.A)
    

    golfBot = DriveBase(left_motor, right_motor, wheel_diameter=26, axle_track=115)
    golfBot = camera_control.DetectedRobot
    
    
          
    
   
