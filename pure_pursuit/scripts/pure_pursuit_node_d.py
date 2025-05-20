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
                ('odom_topic', '/pf/pose/odom'),
                ('drive_topic', '/drive'),
                ('min_lookahead', 0.9),
                ('max_lookahead', 1.2),
                ('min_radius',   1.0),    # m
                ('max_radius',  10.0),    # m
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
        self.is_debug = self.get_parameter("is_debug").get_parameter_value().bool_value  # Debug mode
        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value  # Odom topic
        self.drive_topic = self.get_parameter("drive_topic").get_parameter_value().string_value
        self.min_lookahead = self.get_parameter("min_lookahead").get_parameter_value().double_value
        self.max_lookahead = self.get_parameter("max_lookahead").get_parameter_value().double_value
        self.min_radius = self.get_parameter("min_radius").get_parameter_value().double_value
        self.max_radius = self.get_parameter("max_radius").get_parameter_value().double_value

        self.current_speed = 0.0  # Current speed

        self.waypoints = []  # List of waypoints

        if self.sim:
            self.get_logger().info("Sim mode")
            self.create_subscription(Odometry, self.odom_topic, self.pose_callback, 10)
        else:
            self.create_subscription(Odometry, '/pf/pose/odom', self.pose_callback, 10)


        self.drive_publisher = self.create_publisher(AckermannDriveStamped, self.drive_topic if self.sim else '/drive', 10)

        package_share_directory = get_package_share_directory('pure_pursuit')
        waypoint_file_path = package_share_directory + self.waypoint_file_path_in_package
        
        self.load_waypoints(waypoint_file_path)  # Load waypoints from CSV file

        self.smooth_waypoints()

        self.marker_array = MarkerArray()

        if self.map_sim:
            self.marker_publisher = self.create_publisher(MarkerArray, '/graph_visualization', 10)
            self.publish_waypoints_markers()

        # -------------- NEW: publishes the look‑ahead circle + point -----------------------------
        self.lookahead_pub = self.create_publisher(MarkerArray, '/lookahead_viz', 10)

    def publish_lookahead_visualisation(self, car_x, car_y, lookahead_distance, goal_x, goal_y):
        """
        Publish two markers:
            • a LINE_STRIP circle with radius = lookahead_distance
            • a SPHERE at the look‑ahead point on the path
        Topic: /lookahead_viz   (MarkerArray)
        """
        marker_array = MarkerArray()

        # ------------- circle -------------
        circle = Marker()
        circle.header.frame_id = "map"
        circle.header.stamp = self.get_clock().now().to_msg()
        circle.ns = "lookahead"
        circle.id = 0
        circle.type = Marker.LINE_STRIP
        circle.action = Marker.ADD
        circle.scale.x = 0.03                  # line width
        circle.color.r = 0.0
        circle.color.g = 0.5
        circle.color.b = 1.0
        circle.color.a = 1.0
        circle.pose.orientation.w = 1.0

        # approximate circle with 36 points
        for k in range(37):
            ang = 2*np.pi * k / 36
            p = Point()
            p.x = car_x + lookahead_distance * np.cos(ang)
            p.y = car_y + lookahead_distance * np.sin(ang)
            p.z = 0.0
            circle.points.append(p)
        marker_array.markers.append(circle)

        # ------------- look‑ahead point -------------
        lap = Marker()
        lap.header.frame_id = "map"
        lap.header.stamp = circle.header.stamp
        lap.ns = "lookahead"
        lap.id = 1
        lap.type = Marker.SPHERE
        lap.action = Marker.ADD
        lap.pose.position = Point(x=goal_x, y=goal_y, z=0.0)
        lap.pose.orientation.w = 1.0
        lap.scale.x = lap.scale.y = lap.scale.z = 0.22
        lap.color.r = 1.0
        lap.color.g = 0.0
        lap.color.b = 0.0
        lap.color.a = 1.0
        marker_array.markers.append(lap)

        # publish
        self.lookahead_pub.publish(marker_array)

#       # -------------- END: publishes the look‑ahead circle + point -----------------------------

    def smooth_waypoints(self):
        self.waypoints = np.array(self.waypoints)
        x = self.waypoints[:, 0]
        y = self.waypoints[:, 1]
        t = np.linspace(0, 1, len(x))

        # 2D B-Spline
        tck, u = splprep([x, y], s=8.0, k=2, per=True)  # k=3 for cubic spline

        t_smooth = np.linspace(0, 1, len(x))
        x_smooth, y_smooth = splev(t_smooth, tck)

        self.waypoints = np.vstack((x_smooth, y_smooth)).T

        self.waypoints = self.waypoints.tolist()

    def pose_callback(self, pose_msg):
        pose = pose_msg.pose.pose
        # Dynamically adjust the lookahead distance based on speed or other factors
        # self.adjust_lookahead_distance()

        # --- adaptive look‑ahead -----------------------------------------------
        κ = self.compute_path_curvature(pose)          # new helper
        self.adjust_lookahead_distance(κ)              # updates self.lookahead_distance
        # --- end: adaptive look‑ahead ------------------------------------


        # self._logger.info(f'Lookahead distance: {self.lookahead_distance}')

        # Find the current waypoint to track
        current_waypoint = self.find_current_waypoint(pose)

        # current_position = (pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y)
        # current_yaw = self.get_yaw_from_quaternion(pose_msg.pose.pose.orientation)


        if current_waypoint is not None:

            # ---New: viz: look‑ahead circle & point in global frame ------------------------
            car_x = pose.position.x
            car_y = pose.position.y
            self.publish_lookahead_visualisation(
                car_x, car_y,
                self.lookahead_distance,
                current_waypoint[0], current_waypoint[1])
            # ---End: viz: look‑ahead circle & point in global frame ------------------------



            # Transform goal point to vehicle frame of reference
            goal_point_vehicle_frame = self.transform_to_vehicle_frame(pose, current_waypoint)

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
    

    def compute_path_curvature(self, pose, sample_size=5):
        """
        Three‑point estimate of path curvature κ (1/m) around the car.
        """
        car_xy = np.array([pose.position.x, pose.position.y])

        # nearest waypoint index
        wp = np.array(self.waypoints)
        dists = np.linalg.norm(wp - car_xy, axis=1)
        idx0  = int(np.argmin(dists))

        # take a short arc ahead
        pts = [wp[(idx0 + i) % len(wp)] for i in range(sample_size)]
        if len(pts) < 3:
            return 0.0
        p1, p2, p3 = pts[0], pts[len(pts)//2], pts[-1]

        # triangle area (×2)   (shoelace formula)
        area2 = p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])
        if abs(area2) < 1e-6:
            return 0.0

        # side lengths
        a = np.linalg.norm(p2-p3)
        b = np.linalg.norm(p1-p3)
        c = np.linalg.norm(p1-p2)
        s = 0.5*(a+b+c)
        tri_area = max(1e-6, np.sqrt(s*(s-a)*(s-b)*(s-c)))
        R = (a*b*c)/(4*tri_area)          # circumscribed circle radius
        return 1.0/R

    
    def adjust_lookahead_distance(self, curvature):
        # Adjust the lookahead distance dynamically
        # Example: Adjust based on speed (this is a placeholder, replace with actual logic)
        current_speed = self.current_speed  # Placeholder for current speed
        self._logger.info(f'Current speed: {current_speed}')
        # self.lookahead_distance = max(0.9, min(2.0, current_speed * 0.1))
        # self._logger.info(f'Lookahead distance: {self.lookahead_distance}')

        """
        Map path curvature → look‑ahead distance.
        Tighter curve → shorter preview, straight → longer.
        """
        if abs(curvature) < 1e-6:
            radius = self.max_radius
        else:
            radius = min(max(1.0/abs(curvature), self.min_radius), self.max_radius)

        t = (radius - self.min_radius) / (self.max_radius - self.min_radius)
        self.lookahead_distance = (
            self.min_lookahead + t * (self.max_lookahead - self.min_lookahead)
        )
        # debug
        self._logger.info(
            f"κ={curvature:.3f}  R={radius:.2f}  Ld={self.lookahead_distance:.2f}")

    def find_current_waypoint(self, pose):
        car_x = pose.position.x
        car_y = pose.position.y

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


    def transform_to_vehicle_frame(self, pose, waypoint):
        # Implement transformation from global frame to vehicle frame
        dx = waypoint[0] - pose.position.x
        dy = waypoint[1] - pose.position.y
        yaw = self.get_yaw_from_pose_only(pose) 
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

    # def get_yaw_from_pose(self, pose_msg):
    #     # Convert quaternion to yaw
    #     orientation = pose_msg.pose.orientation
    #     siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
    #     cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
    #     return np.arctan2(siny_cosp, cosy_cosp)
    

    def get_yaw_from_pose_only(self, pose):
        q = pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
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