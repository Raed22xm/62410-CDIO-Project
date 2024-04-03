from control import motor_control
from vision import camera_control

def main():
    camera_control.main()
    motor_control.control()
    
if __name__ == "__main__":
    main()
 

