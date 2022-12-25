
import argparse
import matplotlib.pyplot as plt
from hybrid_a_star.hybrid_a_star import *
from hybrid_a_star.util import *
import hybrid_a_star.reeds_shepp as rs


def main():      

    parser = argparse.ArgumentParser(
    description='''An experiment for applying Hybrid A* path to Frenet's optimal trajectory.''')
    parser.add_argument('-o', '--obstacle-map-pattern', type=str,
                        choices=['ORIGINAL', 'STRAIGHT_ROAD', 'SQUARE', 'ROOF', 'LARGE_V', 'M'],
                        default='ORIGINAL', help='obstacle map pattern')
    args = parser.parse_args()

    ## create obstacle map
    map_pattern = args.obstacle_map_pattern
    omap = ObstacleMap(map_pattern)

    ## set start point and goal point
    ## [x, y, theta]
    start = omap.start
    goal = omap.goal

    plt.plot(omap.ox, omap.oy, ".k")
    rs.plot_arrow(start.x, start.y, start.yaw, fc='g')
    rs.plot_arrow(goal.x, goal.y, goal.yaw)

    plt.grid(True)
    plt.axis("equal")
    # plt.show()

    print("Running Hybrid A * ...")
    path = hybrid_a_star(start, goal, omap)

    x = path.x_list
    y = path.y_list
    yaw = path.yaw_list

    plt.cla()
    plt.plot(omap.ox, omap.oy, ".k")
    plt.plot(x, y, "-r", label="Hybrid A* path")
    # plt.show()
    plt.pause(1)
    print("Hybrid A * done!")

    print("Running Frenet optimal trajectory...")
    frenet_optimal_trajectory(x, y)
    print("Frenet optimal trajectory done!")
    


if __name__ == '__main__':
    main()
