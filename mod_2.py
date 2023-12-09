import airsim

import numpy as np

# connect to the AirSim simulator 
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

car_controls.steering = 0
car_controls.throttle = 0
car_controls.brake = 0

image_buf = np.zeros((1, 144, 256, 3))
state_buf = np.zeros((1,4))

def get_image():
    image = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
    image1d = np.fromstring(image.image_data_uint8, dtype=np.uint8)
    image_rgb = image1d.reshape(image.height, image.width, 3)
    return image_rgb

track_state = 0

while (True):
    car_state = client.getCarState()

    print(f"car speed: {car_state.speed}")

    pose = client.simGetVehiclePose()

    print(f"position: ({pose.position.x_val}, {pose.position.y_val})")

    if (car_state.speed < 15):
        car_controls.throttle = 1.0
    else:
        car_controls.throttle = 0.0

    if track_state == 0 and abs(pose.position.x_val) > 180:
        track_state = 1
        car_controls.steering = 1

    if track_state == 1 and abs(pose.position.y_val) > 15:
        track_state = 2
        car_controls.steering = 0

    if track_state == 2 and abs(pose.position.y_val) > 240:
        track_state = 3
        car_controls.steering = 1

    if track_state == 3 and abs(pose.position.y_val) > 300:
        track_state = 4
        car_controls.steering = 0

    print(f"steering = {car_controls.steering}, throttle = {car_controls.throttle}")

    client.setCarControls(car_controls)
