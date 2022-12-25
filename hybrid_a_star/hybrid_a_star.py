import numpy as np
import heapq
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from math import *
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from frenet_optimal_trajectory.frenet_optimal_trajectory import *
from hybrid_a_star.util import *
from hybrid_a_star import reeds_shepp as rs

def check_valid(node, config):
    x_id, y_id = node.x_id, node.y_id
    if config.min_x <= x_id <= config.max_x and config.min_y <= y_id <= config.max_y:
        return True

    return False

def check_valid_omap(node, omap):
    if node.x < omap.min_x:
        return False
    elif node.y < omap.min_y:
        return False
    elif node.x >= omap.max_x:
        return False
    elif node.y >= omap.max_y:
        return False

    if omap.obstacle_map[node.x][node.y]:
        return False
    return True

def get_index(node, config):
    
    id = (node.yaw_id - config.min_yaw) * config.x_w * config.y_w + \
         (node.y_id - config.min_y) * config.x_w + \
         (node.x_id - config.min_x)

    if id <= 0:
        print("Failed at get_index()")

    return id

def get_index_omap(node, omap):
    return (node.y - omap.min_y) * omap.x_width + (node.x - omap.min_x)

def cal_cost(node, h, config):
    id = (node.y_id - config.min_y) * config.x_w + (node.x_id - config.min_x)
    if id not in h:
        # collision cost
        return node.cost + 999999999  
    return node.cost + H_COST * h[id].cost

def cal_distance_heuristic(gx, gy, omap):
    goal_node = HNode(round(gx / COORD_RESOLUTION), round(gy / COORD_RESOLUTION), 0.0, -1)
    ox = [iox / COORD_RESOLUTION for iox in omap.ox]
    oy = [ioy / COORD_RESOLUTION for ioy in omap.oy]

    openset, closedset = dict(), dict()
    openset[get_index_omap(goal_node, omap)] = goal_node
    pq = [(0, get_index_omap(goal_node, omap))]

    while True:
        if not pq:
            break
        cost, c_id = heapq.heappop(pq)
        if c_id in openset:
            curr = openset[c_id]
            closedset[c_id] = curr
            openset.pop(c_id)
        else:
            continue
        
        # expand search grid
        for i, _ in enumerate(MOTION):
            # parent: cid
            node = HNode(curr.x + MOTION[i][0], curr.y + MOTION[i][1], curr.cost + MOTION[i][2], c_id)
            n_id = get_index_omap(node, omap)

            if (n_id in closedset) or (not check_valid_omap(node, omap)):
                continue

            if n_id not in openset:
                # new node
                openset[n_id] = node 
                heapq.heappush(pq, (node.cost, get_index_omap(node, omap)))
            else:
                if openset[n_id].cost >= node.cost:
                    # best path so far
                    openset[n_id] = node
                    heapq.heappush(pq, (node.cost, get_index_omap(node, omap)))

    return closedset

def check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
    for i_x, i_y, i_yaw in zip(x_list, y_list, yaw_list):
        cx = i_x + BALL_DIST * cos(i_yaw)
        cy = i_y + BALL_DIST * sin(i_yaw)

        ids = kd_tree.query_ball_point([cx, cy], BALL_R)

        if not ids:
            continue

        if not rectangle_check(i_x, i_y, i_yaw,
                               [ox[i] for i in ids], [oy[i] for i in ids]):
            return False  # collision

    return True  # no collision


def check_no_collision(x_list, y_list, yaw_list, omap, kd_tree):
    for x, y, yaw in zip(x_list, y_list, yaw_list):
        cx = x + BALL_DIST * cos(yaw)
        cy = y + BALL_DIST * sin(yaw)

        ids = kd_tree.query_ball_point([cx, cy], BALL_R)
        if not ids:
            continue
        if not rectangle_check(x, y, yaw, [omap.ox[i] for i in ids], [omap.oy[i] for i in ids]):
            return False  

    return True 

def cal_rs_path_cost(rs_path):
    cost = 0.0
    for length in rs_path.lengths:

        if length >= 0:  # forward
            cost += length
        else:  # back
            cost += abs(length) * BACKWARD_COST

    # switch back penalty
    for i in range(len(rs_path.lengths) - 1):
        # switch back
        if rs_path.lengths[i] * rs_path.lengths[i + 1] < 0.0:
            cost += SWITCH_BACK_COST

    # steer penalty
    for course_type in rs_path.ctypes:
        if course_type != "S":  # curve
            cost += STEER_COST * abs(MAX_STEER)

    # calc steer profile
    n_ctypes = len(rs_path.ctypes)
    u_list = [0.0] * n_ctypes
    for i in range(n_ctypes):
        if rs_path.ctypes[i] == "R":
            u_list[i] = - MAX_STEER
        elif rs_path.ctypes[i] == "L":
            u_list[i] = MAX_STEER

    for i in range(len(rs_path.ctypes) - 1):
        cost += STEER_CHANGE_COST * abs(u_list[i + 1] - u_list[i])

    return cost

def analytic_expansion(curr, goal, omap, obstacle_kdtree):
    start_x = curr.x_list[-1]
    start_y = curr.y_list[-1]
    start_yaw = curr.yaw_list[-1]

    goal_x = goal.x_list[-1]
    goal_y = goal.y_list[-1]
    goal_yaw = goal.yaw_list[-1]

    max_curvature = math.tan(MAX_STEER) / LW
    paths = rs.calc_paths(start_x, start_y, start_yaw,
                         goal_x, goal_y, goal_yaw,
                         max_curvature, step_size=MOTION_RESOLUTION)
    if not paths: 
        return None 
    
    best_path, best = None, None 

    for path in paths:
        if check_no_collision(path.x, path.y, path.yaw, omap, obstacle_kdtree):
        # if check_car_collision(path.x, path.y, path.yaw, omap.ox, omap.oy, obstacle_kdtree):
            cost = cal_rs_path_cost(path)
            if not best or best > cost:
                best = cost
                best_path = path
    return best_path

def update_node_with_analytic_expansion(curr, goal, config, omap, kdtree):
    path = analytic_expansion(curr, goal, omap, kdtree)
    
    if path:
        plt.plot(path.x, path.y)
        f_x = path.x[1:]
        f_y = path.y[1:]
        f_yaw = path.yaw[1:]

        f_cost = curr.cost + cal_rs_path_cost(path)

        f_parent_id = get_index(curr, config)

        fd = []
        for d in path.directions[1:]:
            fd.append(d >= 0)

        f_steer = 0.0
        f_path = RouteNode(curr.x_id, curr.y_id, curr.yaw_id, curr.dir, f_x, f_y, f_yaw, fd, 
                           cost=f_cost, pid=f_parent_id, steer=f_steer)
        return True, f_path

    return False, None

def cal_motion_inputs():
    for steer in np.concatenate((np.linspace(-MAX_STEER, MAX_STEER,
                                                N_STEER), [0.0])):
            for d in [1, -1]:
                yield [steer, d]


def find_next_node(curr, steer, dir, config, omap, kd_tree):
    x, y, yaw = curr.x_list[-1], curr.y_list[-1], curr.yaw_list[-1]

    arc_l = COORD_RESOLUTION * 1.5
    x_list, y_list, yaw_list = [], [], []
    for _ in np.arange(0, arc_l, MOTION_RESOLUTION):
        dis = MOTION_RESOLUTION * dir
        x += dis * cos(yaw)
        y += dis * sin(yaw)
        yaw += rs.pi_2_pi(dis * tan(steer) / LW)  # distance/2
        x_list.append(x)
        y_list.append(y)
        yaw_list.append(yaw)

    if not check_no_collision(x_list, y_list, yaw_list, omap, kd_tree):
        return None

    d = dir == 1
   
    added_cost = 0.0

    if d != curr.dir:
        added_cost += SWITCH_BACK_COST

    # steer penalty
    added_cost += STEER_COST * abs(steer)

    # steer change penalty
    added_cost += STEER_CHANGE_COST * abs(curr.steer - steer)

    cost = curr.cost + added_cost + arc_l

    x_id = round(x / COORD_RESOLUTION)
    y_id = round(y / COORD_RESOLUTION)
    yaw_id = round(yaw / YAW_RESOLUTION)

    node = RouteNode(x_id, y_id, yaw_id, d, x_list,
                y_list, yaw_list, [d],
                pid=get_index(curr, config),
                cost=cost, steer=steer)

    return node

def get_neighbors(curr, config, omap, kd_tree):
    for steer, d in cal_motion_inputs():
        node = find_next_node(curr, steer, d, config, omap, kd_tree)
        if node and check_valid(node, config):
            yield node


def get_final_path(closedset, path):
    reversed_x, reversed_y, reversed_yaw = \
        list(reversed(path.x_list)), list(reversed(path.y_list)), \
        list(reversed(path.yaw_list))
    dir = list(reversed(path.dir_list))
    next_id = path.pid
    final_cost = path.cost

    while next_id:
        n = closedset[next_id]
        reversed_x.extend(list(reversed(n.x_list)))
        reversed_y.extend(list(reversed(n.y_list)))
        reversed_yaw.extend(list(reversed(n.yaw_list)))
        dir.extend(list(reversed(n.dir_list)))

        next_id = n.pid

    reversed_x = list(reversed(reversed_x))
    reversed_y = list(reversed(reversed_y))
    reversed_yaw = list(reversed(reversed_yaw))
    dir = list(reversed(dir))

    # adjust first dir
    dir[0] = dir[1]

    path = Path(reversed_x, reversed_y, reversed_yaw, dir, final_cost)

    return path

def hybrid_a_star(start, goal, omap):
    """ Runs Hybrid A* algorithm.
    start: starting point 
        [x, y, theta]
    goal: goal point
        [x, y, theta]
    omap: opstacle_map 
    """

    tmp_ox, tmp_oy = omap.ox[:], omap.oy[:]
    config = Config(tmp_ox, tmp_oy)


    ## contruct kdtree of obstacle points
    obstacle_kdtree = cKDTree(np.vstack((tmp_ox, tmp_oy)).T)
    

    sxid = round(start.x / COORD_RESOLUTION)
    syid = round(start.y / COORD_RESOLUTION)
    syawid = round(start.yaw / YAW_RESOLUTION)

    gxid = round(goal.x / COORD_RESOLUTION)
    gyid = round(goal.y / COORD_RESOLUTION)
    gyawid = round(goal.yaw / YAW_RESOLUTION)

    start_node = RouteNode(sxid, syid, syawid, True, [start.x], [start.y], [start.yaw], [True], cost=0)
    goal_node = RouteNode(gxid, gyid, gyawid, True, [goal.x], [goal.y], [goal.yaw], [True])
    
    
    openset, closedset = {}, {} 
    
    h = cal_distance_heuristic(goal_node.x_list[-1], goal_node.y_list[-1], omap)
    pq = [] 
    openset[get_index(start_node, config)] = start_node
    heapq.heappush(pq, (cal_cost(start_node, h, config), get_index(start_node, config)))
    final_path = None

    while True:
        if not openset:
            print("Error: Cannot find path, No open set")
            return [], [], []

        _, c_id = heapq.heappop(pq)
        if c_id in openset:
            curr = openset.pop(c_id)
            closedset[c_id] = curr
        else:
            continue

        plt.plot(curr.x_list[-1], curr.y_list[-1], "xc")
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if len(closedset.keys()) % 10 == 0:
            plt.pause(0.001)

        is_updated, final_path = update_node_with_analytic_expansion(curr, goal_node, config, omap, obstacle_kdtree)

        if is_updated:
            print("path found")
            break
        for neighbor in get_neighbors(curr, config, omap, obstacle_kdtree):
            neighbor_id = get_index(neighbor, config)
            if neighbor_id in closedset:
                continue
            if neighbor not in openset \
                    or openset[neighbor_id].cost > neighbor.cost:
                heapq.heappush(pq, (cal_cost(neighbor, h, config), neighbor_id))
                openset[neighbor_id] = neighbor

    
    path = get_final_path(closedset, final_path)
    return path


