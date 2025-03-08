#!/usr/bin/env python3
import math
import os
import rclpy
from rclpy.node import Node
import numpy as np
import atexit
from os.path import expanduser
from time import gmtime, strftime
from numpy import linalg as LA
from nav_msgs.msg import Odometry

home = expanduser('~')
log_dir = os.path.join(home, 'rcws/logs')
os.makedirs(log_dir, exist_ok=True)
file = open(strftime(os.path.join(log_dir, 'wp-%Y-%m-%d-%H-%M-%S'), gmtime()) + '.csv', 'w')

class WaypointsLogger(Node):
    def __init__(self):
        super().__init__('waypoints_logger')
        self.subscription = self.create_subscription(
            Odometry,
            'pf/pose/odom',
            self.save_waypoint,
            10)
        self.subscription  # prevent unused variable warning

    def save_waypoint(self, data):
        quaternion = np.array([data.pose.pose.orientation.x, 
                               data.pose.pose.orientation.y, 
                               data.pose.pose.orientation.z, 
                               data.pose.pose.orientation.w])

        euler = euler_from_quaternion(quaternion)
        speed = LA.norm(np.array([data.twist.twist.linear.x, 
                                  data.twist.twist.linear.y, 
                                  data.twist.twist.linear.z]),2)
        if data.twist.twist.linear.x > 0.:
            self.get_logger().info(f'Linear x: {data.twist.twist.linear.x}')

        file.write('%f, %f, %f, %f\n' % (data.pose.pose.position.x,
                                         data.pose.pose.position.y,
                                         euler[2],
                                         speed))

def shutdown():
    file.close()
    print('Goodbye')


 
def euler_from_quaternion(quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def main(args=None):
    rclpy.init(args=args)
    atexit.register(shutdown)
    print('Saving waypoints...')
    waypoints_logger = WaypointsLogger()

    try:
        rclpy.spin(waypoints_logger)
    except KeyboardInterrupt:
        pass
    finally:
        waypoints_logger.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
