# --------------------------------------------------
# Capture & Write Trajectory
# --------------------------------------------------

import argparse
import csv

import airsim
import keyboard

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-t", "--trajectory_filename", type=str, default="trajectory.csv")
    args = argparser.parse_args()

    trajectory_filename = args.trajectory_filename

    trajectory = []

    client = airsim.CarClient()

    client.confirmConnection()

    while (True):
        car_state = client.getCarState()

        print(f"car speed: {car_state.speed}")

        pose = client.simGetVehiclePose()

        print(f"position: ({pose.position.x_val}, {pose.position.y_val})")

        trajectory.append([pose.position.x_val, pose.position.y_val])

        if keyboard.is_pressed("s"):
            break

    print("writing trajectory to file...")

    with open(trajectory_filename, "w") as fd:
        writer = csv.writer(fd, lineterminator="\n")
        writer.writerow(["world_x", "world_y"])
        writer.writerows(trajectory)
