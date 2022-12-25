import math
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

## MODEL
HYBRID = 1
FRENET = 2
HYBRID_FRENET = 3


## resolution
COORD_RESOLUTION = 1.0
YAW_RESOLUTION = np.deg2rad(5.0)
MOTION_RESOLUTION = 0.1

## car 
LW = 3.0  # distance from rear to front wheel
LF = 3.3  # distance from rear to vehicle front end
LB = 1.0  # distance from rear to vehicle back end
W = 2.0  # width of car
MAX_STEER = 0.6  # [rad] maximum steering angle

BALL_DIST = (LF + LB) / 2.0 - LB  # distance from rear to center of vehicle
BALL_R = np.hypot((LF + LB) / 2.0, W / 2.0)  # ball radius


# vehicle rectangle vertices - 4 lines - 5 points
VRX = [LF, LF, -LB, -LB, LF]
VRY = [W / 2, -W / 2, -W / 2, W / 2, W / 2]

MOTION = [[1, 0, 1],
          [0, 1, 1],
          [-1, 0, 1],
          [0, -1, 1],
          [-1, -1, math.sqrt(2)],
          [-1, 1, math.sqrt(2)],
          [1, -1, math.sqrt(2)],
          [1, 1, math.sqrt(2)]]

## cost
H_COST = 5.0  
BACKWARD_COST = 50.0 
SWITCH_BACK_COST = 100.0  
STEER_COST = 1.0 
STEER_CHANGE_COST = 5.0 


N_STEER = 20  # number of steer command



def rectangle_check(x, y, yaw, ox, oy):
    rot = Rotation.from_euler('z', yaw).as_matrix()[0:2, 0:2]
    for iox, ioy in zip(ox, oy):
        tx = iox - x
        ty = ioy - y
        converted_xy = np.stack([tx, ty]).T @ rot
        rx, ry = converted_xy[0], converted_xy[1]

        if not (rx > LF or rx < -LB or ry > W / 2.0 or ry < -W / 2.0):
            return False  
    return True 


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """Plot arrow."""
    if not isinstance(x, float):
        for (i_x, i_y, i_yaw) in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width, alpha=0.4)


def plot_car(x, y, yaw):
    car_color = '-k'
    c, s = math.cos(yaw), math.sin(yaw)
    rot = Rotation.from_euler('z', -yaw).as_matrix()[0:2, 0:2]
    car_outline_x, car_outline_y = [], []
    for rx, ry in zip(VRX, VRY):
        converted_xy = np.stack([rx, ry]).T @ rot
        car_outline_x.append(converted_xy[0]+x)
        car_outline_y.append(converted_xy[1]+y)

    arrow_x, arrow_y, arrow_yaw = c * 1.5 + x, s * 1.5 + y, yaw
    plot_arrow(arrow_x, arrow_y, arrow_yaw)

    plt.plot(car_outline_x, car_outline_y, car_color)

class Config:

    def __init__(self, ox, oy):
        min_x_m = min(ox)
        min_y_m = min(oy)
        max_x_m = max(ox)
        max_y_m = max(oy)

        ox.append(min_x_m)
        oy.append(min_y_m)
        ox.append(max_x_m)
        oy.append(max_y_m)

        self.min_x = round(min_x_m / COORD_RESOLUTION)
        self.min_y = round(min_y_m / COORD_RESOLUTION)
        self.max_x = round(max_x_m / COORD_RESOLUTION)
        self.max_y = round(max_y_m / COORD_RESOLUTION)

        self.x_w = round(self.max_x - self.min_x)
        self.y_w = round(self.max_y - self.min_y)

        self.min_yaw = round(- math.pi / YAW_RESOLUTION) - 1
        self.max_yaw = round(math.pi / YAW_RESOLUTION)
        self.yaw_w = round(self.max_yaw - self.min_yaw)


class SimpleNode:
    """Simple Hybrid A* node."""
    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw

class HNode:
    """Node for calculating heuristic."""
    def __init__(self, x, y, cost, pid):
        self.x = x
        self.y = y
        self.cost = cost
        self.pid = pid

class RouteNode:
    def __init__(self, x_id, y_id, yaw_id, dir,
                 x_list, y_list, yaw_list, dir_list, 
                 steer=0.0, pid=None, cost=None,):
        self.x_id = x_id
        self.y_id = y_id
        self.yaw_id = yaw_id
        self.dir = dir
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.dir_list = dir_list
        self.steer = steer
        self.pid = pid
        self.cost = cost


class Path:

    def __init__(self, x_list, y_list, yaw_list, dir_list, cost):
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.dir_list = dir_list
        self.cost = cost

class ObstacleMap:

    def __init__(self, PATTERN):
        self.ox = self.design_obstacle_map(PATTERN)[0]
        self.oy = self.design_obstacle_map(PATTERN)[1]
        self.cx = self.design_obstacle_map(PATTERN)[2]
        self.cy = self.design_obstacle_map(PATTERN)[3]
        self.start = self.design_obstacle_map(PATTERN)[4]
        self.goal = self.design_obstacle_map(PATTERN)[5]
        self.min_x = None 
        self.min_y = None
        self.max_x = None
        self.max_y = None
        self.x_width = None
        self.y_width = None
        self.get_max()
        self.obstacle_map = self.cal_obstacle_map()

    def get_max(self):
        self.min_x = round(min(self.ox))
        self.min_y = round(min(self.oy))
        self.max_x = round(max(self.ox))
        self.max_y = round(max(self.oy))
        self.x_width = round(self.max_x - self.min_x)
        self.y_width = round(self.max_y - self.min_y)

    def cal_obstacle_map(self):
        obstacle_map = [[False for _ in range(self.y_width)] for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = ix + self.min_x
            for iy in range(self.y_width):
                y = iy + self.min_y
                for iox, ioy in zip(self.ox, self.oy):
                    d = math.sqrt((iox - x) ** 2 + (ioy - y) ** 2)
                    if d <= BALL_R / COORD_RESOLUTION:
                        obstacle_map[ix][iy] = True
                        break

        return obstacle_map
        

    def design_obstacle_map(self, PATTERN):
        ox, oy = [], []
        cx, cy = [], []
        start, goal = None, None

        if PATTERN == 'ORIGINAL':
            # for i in range(60):
            #     ox.append(i)
            #     oy.append(0.0)
            # for i in range(60):
            #     ox.append(60.0)
            #     oy.append(i)
            # for i in range(61):
            #     ox.append(i)
            #     oy.append(60.0)
            # for i in range(61):
            #     ox.append(0.0)
            #     oy.append(i)
            # for i in range(40):
            #     ox.append(20.0)
            #     oy.append(i)
            # for i in range(40):
            #     ox.append(40.0)
            #     oy.append(60.0 - i)

            # # start and gaol 
            # start = SimpleNode(10.0, 10.0, np.deg2rad(90.0))
            # goal = SimpleNode(50.0, 50.0, np.deg2rad(-90.0))
            for i in range(21):
                ox.append(i)
                oy.append(0.0)
            for i in range(60):
                ox.append(20.0)
                oy.append(i)
            for i in range(21):
                ox.append(i)
                oy.append(60.0)
            for i in range(61):
                ox.append(0.0)
                oy.append(i)
            for i in range(40):
                ox.append(5.0)
                oy.append(i)
            for i in range(40):
                ox.append(12.5)
                oy.append(60.0 - i)

            # start and gaol 
            start = SimpleNode(2.50, 10.0, np.deg2rad(90.0))
            goal = SimpleNode(17.5, 50.0, np.deg2rad(90.0))
            

        if PATTERN == 'STRAIGHT_ROAD':
            # Side of road
            for i in range(61):
                ox.append(i)
                oy.append(0.0)
            for i in range(61):
                ox.append(i)
                oy.append(30.0)
            # Different lane
            for i in range(0, 5):
                ox.append(i)
                oy.append(20.0)
            for i in range(10, 20):
                ox.append(i)
                oy.append(20.0)
            for i in range(25, 35):
                ox.append(i)
                oy.append(20.0)
            for i in range(40, 50):
                ox.append(i)
                oy.append(20.0)
            # central 
            for i in range(30):
                cx.append(i)
                cy.append(10)
            for i in range(31, 61):
                cx.append(i)
                cy.append(30)
            
            # start and goal 
            start = SimpleNode(0.0, 15.0, np.deg2rad(0.0))
            goal = SimpleNode(60.0, 25.0, np.deg2rad(0.0))
                
    
        if PATTERN == 'SQUARE':
            for i in range(61):
                ox.append(i)
                oy.append(0.0)
            for i in range(61):
                ox.append(i)
                oy.append(61)
            for i in range(61):
                ox.append(0)
                oy.append(i)
            for i in range(61):
                ox.append(61)
                oy.append(i)

            # central
            for i in range(5, 55):
                cx.append(i)
                cy.append(50)
            for i in range(5, 55):
                cx.append(50)
                cy.append(i)
            
            start = SimpleNode(5.0, 55.0, np.deg2rad(0.0))
            goal = SimpleNode(55.0, 5.0, np.deg2rad(90.0))
            
        
        if PATTERN == 'ROOF':
            for i in range(6):
                ox.append(i)
                oy.append(0)
            for i in range(34, 40):
                ox.append(i)
                oy.append(-5)
            for i in range(20):
                ox.append(i)
                oy.append(i)
            for i in range(6, 20):
                ox.append(i)
                oy.append(i-6)
            for i in range(20, 40):
                ox.append(i)
                oy.append(40-i)
            for i in range(20, 34):
                ox.append(i)
                oy.append(34-i)
            for i in range(-5, 0):
                ox.append(34)
                oy.append(i)
            for i in range(-5, 0):
                ox.append(40)
                oy.append(i)
            
            for i in range(20):
                cx.append(i)
                cy.append(i - 3)
            for i in range(20, 37):
                cx.append(i)
                cy.append(37-i)

            start = SimpleNode(5, 2, np.deg2rad(45.0))
            goal = SimpleNode(36, 2, np.deg2rad(-45.0))
        
        if PATTERN == 'LARGE_V':
            for i in range(10):
                ox.append(i)
                oy.append(0)
            for i in range(30, 40):
                ox.append(i)
                oy.append(-10)
            for i in range(20):
                ox.append(i)
                oy.append(i)
            for i in range(10, 20):
                ox.append(i)
                oy.append(i-10)
            for i in range(20, 40):
                ox.append(i)
                oy.append(40-i)
            for i in range(20, 30):
                ox.append(i)
                oy.append(30-i)
            for i in range(-10, 0):
                ox.append(30)
                oy.append(i)
            for i in range(-10, 0):
                ox.append(40)
                oy.append(i)
                
            for i in range(20):
                cx.append(i)
                cy.append(i - 5)
            for i in range(20, 41):
                cx.append(i)
                cy.append(41-i)

            start = SimpleNode(5, 1.5, np.deg2rad(45.0))
            goal = SimpleNode(35, -7, np.deg2rad(-45.0))

        if PATTERN == 'M':
            for i in range(-5, 1):
                ox.append(i)
                oy.append(-5)
            for i in range(73, 79):
                ox.append(i)
                oy.append(-5)
            for i in range(-5, 20):
                ox.append(i)
                oy.append(i)
            for i in range(1, 20):
                ox.append(i)
                oy.append(i-6)
            for i in range(37, 54):
                ox.append(i)
                oy.append(i-34)
            for i in range(37, 54):
                ox.append(i)
                oy.append(i-40)
                
            for i in range(54, 79):
                ox.append(i)
                oy.append(74-i)
            for i in range(54, 73):
                ox.append(i)
                oy.append(68-i)
            for i in range(20, 37):
                ox.append(i)
                oy.append(40-i)
            for i in range(20, 37):
                ox.append(i)
                oy.append(34-i)

            start = SimpleNode(5, 2, np.deg2rad(45.0))
            goal = SimpleNode(70, 2, np.deg2rad(-45.0))

        return ox, oy, cx, cy, start, goal

