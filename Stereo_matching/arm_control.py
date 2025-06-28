import numpy as np
import time
import pyrealsense2 as rs
import tool as tl
import sympy as sp
import math
from typing import List, Tuple

def is_position_unique(existing_points: List[Tuple[float, float, float, float, float, float]],
                      new_point: Tuple[float, float, float, float, float, float],
                      threshold: float = 15.0) -> bool:

    x_new, y_new, z_new, _, _, _ = new_point
    for idx, (x, y, z, _, _, _) in enumerate(existing_points):
        distance = math.sqrt((x - x_new)**2 + (y - y_new)**2 + (z - z_new)**2)
        if distance < threshold:
            return False
    return True

def calculate_distance_to_origin(x, y, z):
    distance = math.sqrt(x**2 + y**2 + z**2)
    dis2 = math.sqrt(x**2 + y**2)
    return distance,dis2

def base2angle(robot,xyz_position,tool, user, angle_v, speed):
    robot.MoveCart(xyz_position, tool, user, vel=speed)
    print('type1:',xyz_position)
    J = robot.GetActualJointPosDegree()[1]
    J[-1] += angle_v
    robot.MoveJ(J, tool, user, vel=speed)
    path = robot.GetActualTCPPose()[1]
    return path

def move_straight_line(robot,a, b, tool=0, user=0, speed=10, num_steps=100):

    a = np.array(a)
    b = np.array(b)


    for t in np.linspace(0, 1, num_steps):
        current_pos = a + t * (b - a)
        print(t,current_pos)
        robot.MoveCart(current_pos.tolist(), tool, user, vel=speed)
        time.sleep(0.1)  # 每步的时间间隔


def xy_dis2base(depth_intrin, depth_pixel, dis,camer_shift,init_pose,init_path):
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)

    camera_coordinate.append(1.)
    target_list = np.asarray(camera_coordinate).dot(camer_shift.T)

    tcp_base_r_t_matrixs = tl.getRotationAndTransferMatrix(init_pose)
    diagonal_matrix = sp.diag(1, 1, 1)
    r_t_matrix = np.c_[
        np.r_[diagonal_matrix, np.array([[0, 0, 0]], dtype=float)],
        np.array([[target_list[0], target_list[1], target_list[2] - 200, 1.0]], dtype=float).T]
    tcp_base_r_t_matrix = tcp_base_r_t_matrixs
    r = np.array(tcp_base_r_t_matrix.dot(r_t_matrix))
    target_position = r[:, 3]

    target_position_M = [target_position[0],
                         target_position[1],
                         target_position[2]+15] + init_path[-3:]
    return target_position_M
