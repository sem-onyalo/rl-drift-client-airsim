import math
import time
from collections import deque

import airsim
import numpy as np
import pandas as pd
from location import Location
from velocity import Velocity

class Environment():
    def __init__(self, client:airsim.CarClient, trajectory_file_path:str):
        self.client = client

        self.tg = 0
        self.e_vx = 0
        self.e_dis = 0
        self.e_d_vx = 0
        self.away = False
        self.k_heading = 0.1
        self.last_steering = 0.
        self.last_throttle = 0.
        self.destinationFlag = False
        self.controls = self.get_action()

        self.steer_history = deque(maxlen=20)
        self.throttle_history = deque(maxlen=20)
        self.waypoints_history = deque(maxlen=5)

        self.refreshRoute(trajectory_file_path)

    def refreshRoute(self, trajectory_file_path:str):
        traj = pd.read_csv(trajectory_file_path)
        self.route = traj.values
        self.route_x = self.route[:,0]
        self.route_y = self.route[:,1]
        self.route_length = np.zeros(self.route.shape[0])
        for i in range(1, self.route.shape[0]):
            dx = self.route_x[i-1] - self.route_x[i]
            dy = self.route_y[i-1] - self.route_y[i]
            self.route_length[i] = self.route_length[i-1] + np.sqrt(dx * dx + dy * dy)

    def reset(self,
              traj_num=0,
              vehicle_tire_friction=3.5,
              vehicle_mass=1800.0,
              vehicle_number=1,
              vehicle_color="0,0,0",
              vehicle_name=None):
        self.tire_friction = vehicle_tire_friction
        self.mass = vehicle_mass

        self.world.spawn_actor(vehicle_number, vehicle_tire_friction, vehicle_mass, vehicle_color)

        # velocity_local = [10,0]  # 5m/s
        # angular_velocity = carla.Vector3D()
        # ego_yaw = self.start_point.rotation.yaw

        # if not self.collectFlag:
        #     if traj_num not in self.traj_drawn_list:
        #         self.drawPoints()
        #         self.traj_drawn_list.append(traj_num)

        # ego_yaw = ego_yaw/180.0 * 3.141592653
        # transformed_world_velocity = self.velocity_local2world(velocity_local, ego_yaw)

        self.world.player.set_transform(self.start_point)
        # self.world.player.set_target_velocity(transformed_world_velocity)
        # self.world.player.set_target_angular_velocity(angular_velocity)
        self.force_player_stop()
        # self.hud.set_vehicle_name(vehicle_name)
        # self.hud.reset_race_stats()

        self.world.collision_sensor.history = []
        self.away = False
        self.endFlag = False
        self.steer_history.clear()
        self.throttle_history.clear()
        self.waypoints_neighbor = []
        self.waypoints_ahead = []

        self.waypoints_ahead_local = [] # carla.location 10pts
        self.waypoints_history.clear()  # carla.location  5pts
        self.waypoints_history_local = []
        self.destinationFlag = False

        self.last_steering = 0.
        self.last_throttle = 0.

        self.drived_distance = 0

        return 0

    def step(self, steering:float=0., throttle:float=0.):
        controls = self.get_action(steering=steering,throttle=throttle)
        controls.steering = 0.1*controls.steering + 0.9*self.last_steering
        controls.throttle = 0.3*controls.throttle + 0.7*self.last_throttle
        self.controls = controls

        self.last_steering = controls.steering
        self.last_throttle = controls.throttle

        self.client.setCarControls(controls)
        self.steer_history.append(controls.steering)
        self.throttle_history.append(controls.throttle)
        time.sleep(0.05)

        new_state = self.get_state()

        reward = self.get_reward(new_state, self.steer_history, self.throttle_history)

        collisionFlag = self.collision_detect()

        return new_state, reward, collisionFlag, self.destinationFlag, self.away, self.controls

    def get_action(self, steering:float=0., throttle:float=0.):
        controls = airsim.CarControls()
        controls.steering = steering
        controls.throttle = throttle
        return controls

    def get_state(self):
        location = self.get_location()

        ego_yaw = location.yaw
        if ego_yaw < 0:
            ego_yaw += 360
        if ego_yaw > 360:
            ego_yaw -= 360
        ego_yaw = ego_yaw/180.0 * 3.141592653

        self.getNearby() # will update self.minDis

        self.getLocalHistoryWay(location, ego_yaw)

        self.getLocalFutureWay(location, ego_yaw)

        self.velocity_world2local(ego_yaw) # will update self.velocity_local

        ego_yaw = ego_yaw/3.141592653 * 180
        if ego_yaw > 180:
            ego_yaw = -(360-ego_yaw)

        dt = time.time() - self.tg
        self.e_d_dis = (self.minDis - self.e_dis) / dt
        self.e_dis = self.minDis

        if self.e_dis > 15:
            self.away = True

        # error of heading:
        # 1. calculate the abs
        way_yaw = self.waypoints_ahead[0,2]
        # 2. update the way_yaw based on vector guidance field:
        vgf_left = self.vgf_direction(location)  
        # 3. if the vehicle is on the left of the nearst waypoint, according to the heading of the waypoint
        if vgf_left:
            way_yaw = math.atan(self.k_heading * self.e_dis)/3.141592653*180 + way_yaw
        else:
            way_yaw = -math.atan(self.k_heading * self.e_dis)/3.141592653*180 + way_yaw
        if way_yaw > 180:
            way_yaw = -(360-way_yaw)
        if way_yaw < -180:
            way_yaw += 360

        if ego_yaw*way_yaw > 0:
            e_heading = abs(ego_yaw - way_yaw)
        else:
            e_heading = abs(ego_yaw) + abs(way_yaw)
            if e_heading > 180:
                e_heading = 360 - e_heading
        # considering the +-:
        # waypoint to the vehicle, if clockwise, then +
        hflag = 1
        if ego_yaw*way_yaw > 0:
            if ego_yaw > 0:
                if abs(way_yaw) < abs(ego_yaw):
                    hflag = -1
                else:
                    hflag = 1
            if ego_yaw < 0:
                if abs(way_yaw) < abs(ego_yaw):
                    hflag = 1
                else:
                    hflag = -1
        else:
            if ego_yaw > 0:
                t_yaw = ego_yaw-180
                if way_yaw > t_yaw:
                    hflag = -1
                else:
                    hflag = 1
            else:
                t_yaw = ego_yaw + 180
                if way_yaw > t_yaw:
                    hflag = -1
                else:
                    hflag = 1
        e_heading = e_heading * hflag
        if e_heading * self.e_heading > 0:
            if e_heading > 0:
                self.e_d_heading = (e_heading - self.e_heading)/dt
            else:
                self.e_d_heading = -(e_heading - self.e_heading)/dt
        else:
            self.e_d_heading = (abs(e_heading) - abs(self.e_heading)) / dt

        self.e_heading = e_heading

        # e_slip = self.velocity_local[2] - self.waypoints_ahead[0,5]
        # self.e_d_slip = (e_slip - self.e_slip)/dt
        # self.e_slip = e_slip

        # 30.56 == hard-code to 110 km/h
        e_vx = self.velocity_local[0] - 30.56 # self.waypoints_ahead[0,3]
        self.e_d_vx = (e_vx - self.e_vx)/dt
        self.e_vx = e_vx

        # e_vy = self.velocity_local[1] - self.waypoints_ahead[0,4]
        # self.e_d_vy = (e_vy - self.e_vy)/dt
        # self.e_vy = e_vy

        # self.control = self.world.player.get_control()

        steering = self.controls.steering
        throttle = self.controls.throttle

        ct = time.time()
        if ct - self.clock_history > 0.2:
            self.waypoints_history.append(np.array([location.x, location.y, steering, self.velocity_local[2]]))
            self.clock_history = ct

        # vx = self.velocity_local[0]
        # vy = self.velocity_local[1]
        # e_d_slip = self.e_d_slip
        # if math.sqrt(vx*vx + vy*vy) < 2: # if the speed is too small we ignore the error of slip angle
        #     e_slip = 0
        #     e_d_slip = 0

        state = [
            steering,
            throttle,
            self.e_dis,
            self.e_d_dis,
            self.e_heading,
            self.e_d_heading,
            0, # e_slip,
            0, # e_d_slip,
            self.e_vx,
            self.e_d_vx,
            0, # self.e_vy,
            0, # self.e_d_vy
        ]

        # put this here because it wasn't extending the state to the full expected size 
        # which would then go into the NN and cause -> ValueError: cannot reshape array of size 39 into shape (1,42)
        pad_state = len(self.waypoints_ahead_local) < 10
        if pad_state:
            pad_what = [0] * (10 - len(self.waypoints_ahead_local))

        state.extend([k[0] for k in self.waypoints_ahead_local]) #x
        if pad_state:
            state.extend(pad_what)

        state.extend([k[1] for k in self.waypoints_ahead_local]) #y
        if pad_state:
            state.extend(pad_what)

        state.extend([k[2] for k in self.waypoints_ahead_local]) #slip
        if pad_state:
            state.extend(pad_what)

        self.tg = time.time()

        return state

    def get_location(self) -> Location:
        pose = self.client.simGetVehiclePose()
        quat = pose.orientation
        yaw = math.atan2(2.*(quat.y_val*quat.z_val + quat.w_val*quat.x_val), quat.w_val*quat.w_val - quat.x_val*quat.x_val - quat.y_val*quat.y_val + quat.z_val*quat.z_val)
        return Location(pose.position.x_val, pose.position.y_val, yaw)

    def get_velocity(self) -> Velocity:
        gps_data = self.client.getGpsData()
        return Velocity(gps_data.gnss.velocity.x_val, gps_data.gnss.velocity.y_val)

    def getNearby(self):
        self.waypoints_ahead = [] 
        self.waypoints_neighbor = []
        egoLocation = self.get_location()
        dx_array = self.route_x - egoLocation.x
        dy_array = self.route_y - egoLocation.y
        dis_array = np.sqrt(dx_array * dx_array + dy_array * dy_array)
        self.minDis = np.amin(dis_array)
        _ = np.where(dis_array == self.minDis)
        index = _[0][0]  # index for the min distance to all waypoints.

        self.drived_distance = self.route_length[index]
        self.waypoints_ahead = self.route[index:,:]

        if index >= 20:
            index_st = index - 20
        else:
            index_st = 0
        self.waypoints_neighbor = self.route[index_st:,:]
        self.traj_index = index

    def getLocalHistoryWay(self, egoLocation, yaw):
        # x, y, steer, slip (degree)
        ways = self.waypoints_history
        yaw = -yaw
        self.waypoints_history_local = []
        if len(ways) < 5:
            for i in range(5 - len(ways)):
                self.waypoints_history_local.append(np.array([0,0,0,0]))

        for w in ways:
            wx = w[0]
            wy = w[1]
            w_steer = w[2]
            w_slip = w[3]
            dx = wx - egoLocation.x
            dy = wy - egoLocation.y

            nx = dx * math.cos(yaw) - dy * math.sin(yaw)
            ny = dy * math.cos(yaw) + dx * math.sin(yaw)
            self.waypoints_history_local.append(np.array([nx,ny,w_steer,w_slip]))

    def getLocalFutureWay(self, egoLocation, yaw):
        # transfer the future waypoints (#10) to the local coordinate.
        # x, y, slip (degree)

        ways = self.waypoints_ahead[0:-1:5,:]  # filter to 1m between way pts
        # close_to_finish_line = False
        if ways.shape[0] <= 10:
            ways = self.waypoints_ahead
            # close_to_finish_line = True
        if ways.shape[0] == 0: #  < 11:
            self.destinationFlag = True

        # if close_to_finish_line:
        #     pos_thsh = 1.5
        #     check_diff_x = True
        #     location = self.world.player.get_location()
        #     pos_ends = ways[-1:][0][0] if check_diff_x else ways[-1:][0][1]
        #     pos_curr = location.x if check_diff_x else location.y
        #     pos_diff = abs(pos_curr - pos_ends)
        #     if pos_diff < pos_thsh:
        #         self.world.hud.stop_race_timer()
        #         self.destinationFlag = False

        self.waypoints_ahead_local = []
        yaw = -yaw

        for w in ways[0:10]: 
            wx = w[0]
            wy = w[1]
            w_slip = 0 # w[5]
            dx = wx - egoLocation.x
            dy = wy - egoLocation.y

            nx = dx * math.cos(yaw) - dy * math.sin(yaw)
            ny = dy * math.cos(yaw) + dx * math.sin(yaw)
            self.waypoints_ahead_local.append(np.array([nx, ny, w_slip]))

    def velocity_world2local(self, yaw):
        velocity_world = self.get_velocity()
        vx = velocity_world.x
        vy = velocity_world.y
        yaw = -yaw

        local_x = float(vx * math.cos(yaw) - vy * math.sin(yaw))
        local_y = float(vy * math.cos(yaw) + vx * math.sin(yaw))
        if local_x != 0:
            slip_angle = math.atan(local_y/local_x)/3.1415926*180
        else:
            slip_angle = 0
        
        self.velocity_local = [local_x, local_y, slip_angle]

    def vgf_direction(self, egoLocation):
        way_x = self.waypoints_ahead[0,0]
        way_y = self.waypoints_ahead[0,1]
        yaw = -self.waypoints_ahead[0,2]/180.0 * 3.141592653

        dx = egoLocation.x - way_x
        dy = egoLocation.y - way_y

        # nx = dx * math.cos(yaw) - dy * math.sin(yaw)
        ny = dy * math.cos(yaw) + dx * math.sin(yaw)

        if ny < 0:
            return True
        else:
            return False

    def get_reward(self, state, steer_history, throttle_history):
        e_dis = state[2]
        e_slip = state[6]
        e_heading = state[4]
        std_steer = np.array(steer_history)
        std_steer = std_steer.std()

        std_throttle = np.array(throttle_history)
        std_throttle = std_throttle.std()

        r_dis = np.exp(-0.5*e_dis)

        if abs(e_heading)<90:
            r_heading = np.exp(-0.1*abs(e_heading))
        elif (e_heading)>=90:
            r_heading = -np.exp(-0.1*(180-e_heading))
        else:
            r_heading = -np.exp(-0.1*(e_heading+180))

        if abs(e_slip)<90:
            r_slip = np.exp(-0.1*abs(e_slip))
        elif (e_slip)>= 90:
            r_slip = -np.exp(-0.1*(180-e_slip))
        else:
            r_slip = -np.exp(-0.1*(e_slip+180))

        # r_std_steer = np.exp(-2*std_steer)
        # r_std_throttle = np.exp(-2*std_throttle)

        vx = self.velocity_local[0]
        vy = self.velocity_local[1]
        v = math.sqrt(vx*vx + vy*vy)

        reward = v*(40*r_dis + 40*r_heading + 20*r_slip)

        if v < 6:
            reward = reward / 2

        return reward

    def collision_detect(self):
        car_state = self.client.getCarState()
        return car_state.collision.has_collided
