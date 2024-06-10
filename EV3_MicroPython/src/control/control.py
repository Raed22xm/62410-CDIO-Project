import requests
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port, Stop
from pybricks.tools import wait

# Initialize the EV3 brick.
ev3 = EV3Brick()

# Initialize the motors.
left_motor = Motor(Port.B)
right_motor = Motor(Port.C)

def fetch_data(endpoint):
    url = f'http://10.209.142.154:5000/{endpoint}'  # Replace with your server's IP if it changes
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch {endpoint}")
        return None

def move_forward_and_stop(speed, duration):
    left_motor.run(speed)
    right_motor.run(speed)
    wait(duration)
    left_motor.stop(Stop.HOLD)
    right_motor.stop(Stop.HOLD)
    ev3.speaker.beep()

def main():
    balls = fetch_data('balls')
    robots = fetch_data('robots')
    field = fetch_data('field')
    obstacles = fetch_data('obstacles')
    
    print("Balls:", balls)
    print("Robots:", robots)
    print("Field:", field)
    print("Obstacles:", obstacles)

    # Example usage: move forward at 500 degrees per second for 2 seconds
    move_forward_and_stop(500, 2000)

if __name__ == '__main__':
    main()