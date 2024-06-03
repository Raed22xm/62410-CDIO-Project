from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port, Stop
from pybricks.tools import wait

# Initialize the EV3 brick.
ev3 = EV3Brick()

# Initialize the motors.
left_motor = Motor(Port.B)
right_motor = Motor(Port.C)

def move_forward_and_stop(speed, duration):
    """
    Move the robot forward at the specified speed for the specified duration, then stop.

    Args:
    - speed (int): The speed at which the motors should run (degrees per second).
    - duration (int): The duration for which the motors should run (milliseconds).
    """
    # Set both motors to run at the specified speed
    left_motor.run(speed)
    right_motor.run(speed)
    
    # Wait for the specified duration
    wait(duration)
    
    # Stop both motors
    left_motor.stop(Stop.HOLD)
    right_motor.stop(Stop.HOLD)

    # Optional: beep to indicate that the movement is complete
    ev3.speaker.beep()

# Example usage: move forward at 500 degrees per second for 2 seconds
move_forward_and_stop(500, 2000)
