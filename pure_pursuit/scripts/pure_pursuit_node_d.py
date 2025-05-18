#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import csv
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import MarkerArray, Marker
from ament_index_python.packages import get_package_share_directory
from scipy.interpolate import splprep, splev


class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('pure_pursuit_node_d')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('lookahead_distance', 1.0),
                ('curvature_gain', 0.5),
                ("waypoint_file_path_in_package", "/config/wp-2025-03-08-01-27-47_first_complete_loop.csv"),
                ('max_speed', 3.0),
                ('min_speed', 0.5),
                ('max_steering_angle', np.pi / 4),
                ('sim', True),
                ('map_sim', False),
                ('is_debug', False),
            ]
        )
        self.sim = self.get_parameter("sim").get_parameter_value().bool_value
        self.map_sim = self.get_parameter("map_sim").get_parameter_value().bool_value
        self.lookahead_distance = self.get_parameter("lookahead_distance").get_parameter_value().double_value  # Initial lookahead distance
        self.max_speed = self.get_parameter("max_speed").get_parameter_value().double_value  # Maximum speed
        self.min_speed = self.get_parameter("min_speed").get_parameter_value().double_value  # Minimum speed
        self.max_steering_angle = self.get_parameter("max_steering_angle").get_parameter_value().double_value  # Maximum steering angle
        self.curvature_gain = self.get_parameter("curvature_gain").get_parameter_value().double_value  # Curvature gain
        self.waypoint_file_path_in_package = self.get_parameter("waypoint_file_path_in_package").get_parameter_value().string_value  # Waypoint file path

        self.current_speed = 0.0  # Current speed

        self.waypoints = []  # List of waypoints

        if self.sim:
            self.get_logger().info("Sim mode")
            self.create_subscription(Odometry, '/opp_racecar/odom', self.pose_callback, 10)
        else:
            self.create_subscription(Odometry, '/pf/pose/odom', self.pose_callback, 10)


        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/opp_drive' if self.sim else '/drive', 10)

        package_share_directory = get_package_share_directory('pure_pursuit')
        waypoint_file_path = package_share_directory + self.waypoint_file_path_in_package
        
        self.load_waypoints(waypoint_file_path)  # Load waypoints from CSV file

        self.smooth_waypoints()

        self.marker_array = MarkerArray()

        if self.map_sim:
            self.marker_publisher = self.create_publisher(MarkerArray, '/graph_visualization', 10)
            self.publish_waypoints_markers()

    def smooth_waypoints(self):
        self.waypoints = np.array(self.waypoints)
        x = self.waypoints[:, 0]
        y = self.waypoints[:, 1]
        t = np.linspace(0, 1, len(x))

        # 2D B-Spline
        tck, u = splprep([x, y], s=8.0, k=3, per=True)  # k=3 for cubic spline

        t_smooth = np.linspace(0, 1, len(x))
        x_smooth, y_smooth = splev(t_smooth, tck)

        self.waypoints = np.vstack((x_smooth, y_smooth)).T

        self.waypoints = self.waypoints.tolist()

    def pose_callback(self, pose_msg):
        pose_msg = pose_msg.pose
        # Dynamically adjust the lookahead distance based on speed or other factors
        # self.adjust_lookahead_distance()

        # Find the current waypoint to track
        current_waypoint = self.find_current_waypoint(pose_msg)

        # current_position = (pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y)
        # current_yaw = self.get_yaw_from_quaternion(pose_msg.pose.pose.orientation)


        if current_waypoint is not None:
            # Transform goal point to vehicle frame of reference
            goal_point_vehicle_frame = self.transform_to_vehicle_frame(pose_msg, current_waypoint)

            # Calculate curvature/steering angle
            curvature = self.calculate_curvature(goal_point_vehicle_frame)
            if self.is_debug:
                self.get_logger().info(f'Curvature gain: {self.curvature_gain}')
            
            steering_angle = np.arctan(curvature * self.lookahead_distance * self.curvature_gain)

            # Publish drive message
            self.publish_drive_message(steering_angle)

    def get_yaw_from_quaternion(self, quaternion):
            """
            Convert a quaternion to yaw (rotation around the z-axis).
            """
            x = quaternion.x
            y = quaternion.y
            z = quaternion.z
            w = quaternion.w

            # Calculate yaw
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)

            return yaw
    
    def adjust_lookahead_distance(self):
        # Adjust the lookahead distance dynamically
        # Example: Adjust based on speed (this is a placeholder, replace with actual logic)
        current_speed = self.current_speed  # Placeholder for current speed
        self.lookahead_distance = max(0.9, min(2.0, current_speed * 0.1))

    def find_current_waypoint(self, pose_msg):
        car_x = pose_msg.pose.position.x
        car_y = pose_msg.pose.position.y

        if self.is_debug:
            self.get_logger().info(f'Car position: ({car_x}, {car_y})')

        closest_waypoint = None
        next_waypoint = None
        for i in range(len(self.waypoints) - 1):
            wp1 = self.waypoints[i]
            wp2 = self.waypoints[i + 1]
            dist1 = np.sqrt((wp1[0] - car_x)**2 + (wp1[1] - car_y)**2)
            dist2 = np.sqrt((wp2[0] - car_x)**2 + (wp2[1] - car_y)**2)
            if dist1 < self.lookahead_distance and dist2 > self.lookahead_distance:
                closest_waypoint = wp1
                next_waypoint = wp2
                break

        if closest_waypoint is None or next_waypoint is None:
            return None

        c_w = self.interpolate_waypoints(closest_waypoint, next_waypoint, car_x, car_y)
        
        
        return c_w

    def interpolate_waypoints(self, wp1, wp2, car_x, car_y):
        dx = wp2[0] - wp1[0]
        dy = wp2[1] - wp1[1]
        dist_wp = np.sqrt(dx**2 + dy**2)
        t = (self.lookahead_distance - np.sqrt((wp1[0] - car_x)**2 + (wp1[1] - car_y)**2)) / dist_wp
        goal_x = wp1[0] + t * dx
        goal_y = wp1[1] + t * dy
        return (goal_x, goal_y)
    
    def load_waypoints(self, csv_file_path):
        with open(csv_file_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                x, y = float(row[0]), float(row[1])
                if float(row[3]) != 0:
                    self.waypoints.append((x, y))

    def publish_waypoints_markers(self, current_waypoint=None):
        way_point_copy = self.waypoints.copy()
        self.get_logger().info("Length of waypoints: " + str(len(way_point_copy)))
        if current_waypoint:
            way_point_copy.append(current_waypoint)
        # marker_array_d = MarkerArray()
        # marker_target_way_point = Marker()
        # marker_target_way_point._id = 0
        # marker_target_way_point.action = Marker.DELETE
        # marker_array_d.markers.append(marker_target_way_point)
        # self.marker_publisher.publish(marker_array_d)

        marker_array = self.marker_array

        for i, waypoint in enumerate(way_point_copy):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "waypoints"
            marker.id = i + 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = Point(x=waypoint[0], y=waypoint[1], z=0.0)
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            if current_waypoint and waypoint == current_waypoint:
                marker.scale.x = 0.3
                marker.scale.y = 0.3
                marker.scale.z = 0.3
                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.id = 0
            else:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            marker_array.markers.append(marker)
        self.marker_publisher.publish(marker_array)


    def transform_to_vehicle_frame(self, pose_msg, waypoint):
        # Implement transformation from global frame to vehicle frame
        dx = waypoint[0] - pose_msg.pose.position.x
        dy = waypoint[1] - pose_msg.pose.position.y
        yaw = self.get_yaw_from_pose(pose_msg)
        x_vehicle = dx * np.cos(yaw) + dy * np.sin(yaw)
        y_vehicle = -dx * np.sin(yaw) + dy * np.cos(yaw)
        return (x_vehicle, y_vehicle)

    def calculate_curvature(self, goal_point):
        # Implement curvature calculation
        x, y = goal_point
        return 2 * y / (self.lookahead_distance ** 2)

    def publish_drive_message(self, steering_angle):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle

        # Calculate speed based on the steering angle
        max_speed = self.max_speed  # Maximum speed
        min_speed = self.min_speed  # Minimum speed
        max_steering_angle = self.max_steering_angle  # Maximum steering angle (45 degrees)
        
        # Speed is inversely proportional to the absolute value of the steering angle
        # speed = max_speed - (max_speed - min_speed) * (abs(steering_angle) / max_steering_angle)
        speed = max_speed * (1 - (abs(steering_angle) / max_steering_angle))
        speed = max(min_speed, min(max_speed, speed))  # Ensure speed is within bounds

        drive_msg.drive.speed = speed
        self.current_speed = speed
        self.drive_publisher.publish(drive_msg)

    def get_yaw_from_pose(self, pose_msg):
        # Convert quaternion to yaw
        orientation = pose_msg.pose.orientation
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        return np.arctan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit dummy Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()