# --------------------------------------------------
# Train
# --------------------------------------------------

import argparse
import logging

import airsim
from environment import Environment

__logger = logging.getLogger(f"rl-drift-airsim.trainer")

class RuntimeArgs:
    log_level:str
    trajectory_filename:str

def get_runtime_args() -> RuntimeArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trajectory_filename", type=str, default="trajectory.csv")
    parser.add_argument("-l", "--log-level", type=str, default="INFO")
    return parser.parse_args()

def init_logger(level:str):
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.getLevelName(level.upper())
    )

if __name__ == "__main__":
    args = get_runtime_args()

    init_logger(args.log_level)

    trajectory_filename = args.trajectory_filename

    client = airsim.CarClient()

    client.confirmConnection()

    env = Environment(client, trajectory_filename)

    env.reset()

    while (True):
        car_state = client.getCarState()

        print(f"car speed: {car_state.speed}")

        pose = client.simGetVehiclePose()

        print(f"position: ({pose.position.x_val}, {pose.position.y_val})")
