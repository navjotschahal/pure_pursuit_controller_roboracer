#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import csv
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from geometry_msgs.msg import PoseStamped, Point, TransformStamped
from visualization_msgs.msg import MarkerArray, Marker
from ament_index_python.packages import get_package_share_directory
from scipy.interpolate import splprep, splev
from tf2_ros import TransformBroadcaster
import math
from scipy.spatial.distance import cdist


class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car with obstacle detection and lane switching
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('lookahead_distance', 1.5),
                ('curvature_gain', 0.5),
                ('lane1_file_path', "/config/wp-2025-03-08-01-27-47_first_complete_loop.csv"),
                ('lane2_file_path', "/config/lane2_waypoints.csv"),
                ('lane3_file_path', "/config/lane3_waypoints.csv"),
                ('active_lane', 'lane1'),  # Default lane to follow
                ('max_speed', 3.0),
                ('min_speed', 0.5),
                ('max_steering_angle', np.pi / 4),
                ('sim', True),
                ('map_sim', False),
                # --------------------------------------------------------------------------------
                ('min_lookahead', 0.9),
                ('max_lookahead', 1.2),
                ('min_radius',   1.0),    # m
                ('max_radius',  10.0),    # m
                #--------------------------------------------------------------------------------
                # Obstacle detection parameters
                ('obstacle_detection_distance', 3.0),  # How far ahead to check for obstacles
                ('obstacle_proximity_threshold', 0.3),  # How close to waypoints to consider a lane blocked
                ('scan_topic', '/scan'),  # Topic for laser scan
                ('enable_lane_switching', True)  # Enable automatic lane switching
            ]
        )
        
        # Get existing parameters
        self.sim = self.get_parameter("sim").get_parameter_value().bool_value
        self.map_sim = self.get_parameter("map_sim").get_parameter_value().bool_value
        self.lookahead_distance = self.get_parameter("lookahead_distance").get_parameter_value().double_value
        self.max_speed = self.get_parameter("max_speed").get_parameter_value().double_value
        self.min_speed = self.get_parameter("min_speed").get_parameter_value().double_value
        self.max_steering_angle = self.get_parameter("max_steering_angle").get_parameter_value().double_value
        self.curvature_gain = self.get_parameter("curvature_gain").get_parameter_value().double_value
        self.min_lookahead = self.get_parameter("min_lookahead").get_parameter_value().double_value
        self.max_lookahead = self.get_parameter("max_lookahead").get_parameter_value().double_value
        self.min_radius = self.get_parameter("min_radius").get_parameter_value().double_value
        self.max_radius = self.get_parameter("max_radius").get_parameter_value().double_value
        
        # Get lane file paths
        self.lane1_file_path = self.get_parameter("lane1_file_path").get_parameter_value().string_value
        self.lane2_file_path = self.get_parameter("lane2_file_path").get_parameter_value().string_value
        self.lane3_file_path = self.get_parameter("lane3_file_path").get_parameter_value().string_value
        self.active_lane = self.get_parameter("active_lane").get_parameter_value().string_value
        
        # Obstacle detection parameters
        self.obstacle_detection_distance = self.get_parameter("obstacle_detection_distance").get_parameter_value().double_value
        self.obstacle_proximity_threshold = self.get_parameter("obstacle_proximity_threshold").get_parameter_value().double_value
        self.scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.enable_lane_switching = self.get_parameter("enable_lane_switching").get_parameter_value().bool_value
        
        self.current_speed = 0.0  # Current speed
        self.current_pose = None  # Store current car pose
        self.obstacles = []       # List to store detected obstacles (x, y) in car frame
        self.lane_status = {      # Status of each lane
            'lane1': {'blocked': False, 'distance_to_obstacle': float('inf')},
            'lane2': {'blocked': False, 'distance_to_obstacle': float('inf')},
            'lane3': {'blocked': False, 'distance_to_obstacle': float('inf')}
        }
        self.last_lane_switch_time = self.get_clock().now()  # To prevent too frequent lane switches
        self.lane_switch_cooldown = 0.0  # Seconds between lane switches
        
        # Dictionary to store waypoints for multiple lanes
        self.lanes = {
            'lane1': [],
            'lane2': [],
            'lane3': []
        }

        # Subscribe to pose topics
        if self.sim:
            self.create_subscription(Odometry, '/ego_racecar/odom', self.pose_callback, 10)
        else:
            self.create_subscription(Odometry, '/pf/pose/odom', self.pose_callback, 10)

        # Subscribe to scan topic for obstacle detection
        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)

        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        
        # Create publisher for obstacle visualization
        self.obstacle_marker_publisher = self.create_publisher(MarkerArray, '/obstacle_markers', 10)
        self.obstacle_marker_array = MarkerArray()

        package_share_directory = get_package_share_directory('pure_pursuit')
        
        # Load waypoints for each lane
        self.load_waypoints(package_share_directory + self.lane1_file_path, 'lane1')
        self.load_waypoints(package_share_directory + self.lane2_file_path, 'lane2')
        self.load_waypoints(package_share_directory + self.lane3_file_path, 'lane3')

        # Smooth waypoints for each lane
        for lane in self.lanes:
            self.smooth_waypoints(lane)

        self.marker_array = MarkerArray()

        if self.map_sim:
            self.marker_publisher = self.create_publisher(MarkerArray, '/graph_visualization', 10)
            self.publish_waypoints_markers()
            
        self.get_logger().info(f"Pure pursuit initialized with obstacle detection. Active lane: {self.active_lane}")


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

    def scan_callback(self, msg):
        """Process laser scan data to detect obstacles"""
        if self.current_pose is None:
            return
        
        obstacles = []
        angles = np.arange(msg.angle_min, msg.angle_max + msg.angle_increment, msg.angle_increment)
        
        # Process scan data - filter by range and field of view
        forward_view_angle = np.pi/3  # 60 degrees forward view
        min_angle = -forward_view_angle/2
        max_angle = forward_view_angle/2
        
        for i, (angle, range_val) in enumerate(zip(angles, msg.ranges)):
            # Skip invalid readings or readings not in our forward view
            if (angle < min_angle or angle > max_angle or 
                range_val < msg.range_min or 
                range_val > msg.range_max or range_val == 0 or range_val >= 10 or
                range_val > self.obstacle_detection_distance or
                not np.isfinite(range_val)):
                continue
                
            # Convert from polar to Cartesian coordinates (in vehicle frame)
            x = range_val * np.cos(angle)
            y = range_val * np.sin(angle)
            
            # Only add obstacles in front of the vehicle
            if x > 0:
                obstacles.append((x, y))
        
        # Update obstacles list
        self.obstacles = obstacles
        
        # Check if any lane is blocked
        self.check_lanes_for_obstacles()
        
        # Visualize obstacles
        self.publish_obstacle_markers(obstacles)

    def check_lanes_for_obstacles(self):
        """Determine which lanes are blocked by obstacles"""
        # First reset lane status - moved before the early return!
        for lane in self.lane_status:
            self.lane_status[lane]['blocked'] = False
            self.lane_status[lane]['distance_to_obstacle'] = float('inf')
        
        # Early return if no valid data
        if not self.obstacles or self.current_pose is None:
            # Even with no obstacles, check if we should switch back to preferred lane
            self.check_and_switch_lanes()
            return
        
        # Transform obstacles from vehicle frame to map frame
        car_x = self.current_pose.position.x
        car_y = self.current_pose.position.y
        car_yaw = self.get_yaw_from_pose(self.current_pose.orientation)
        
        # Define a rotation matrix for the vehicle orientation
        cos_yaw = np.cos(car_yaw)
        sin_yaw = np.sin(car_yaw)
        
        obstacles_map_frame = []
        for obs_x, obs_y in self.obstacles:
            # Transform from vehicle frame to map frame
            map_x = car_x + obs_x * cos_yaw - obs_y * sin_yaw
            map_y = car_y + obs_x * sin_yaw + obs_y * cos_yaw
            obstacles_map_frame.append((map_x, map_y))
        
        if not obstacles_map_frame:
            return
            
        # Check each lane
        for lane_name, waypoints in self.lanes.items():
            if not waypoints:
                continue
                
            # Get waypoints ahead of the vehicle within our detection distance
            relevant_waypoints = []
            for wp in waypoints:
                dist_to_car = np.sqrt((wp[0] - car_x)**2 + (wp[1] - car_y)**2)
                # Only check waypoints ahead of us within the detection distance
                if dist_to_car < self.obstacle_detection_distance:
                    angle_to_wp = np.arctan2(wp[1] - car_y, wp[0] - car_x)
                    # Check if waypoint is ahead (within 90 degrees of car's heading)
                    if abs(self.normalize_angle(angle_to_wp - car_yaw)) < np.pi/2:
                        relevant_waypoints.append(wp)
            
            if not relevant_waypoints:
                continue
                
            # Convert relevant waypoints to numpy array for distance calculation
            relevant_waypoints = np.array(relevant_waypoints)
            obstacles_map_frame_array = np.array(obstacles_map_frame)
            
            # Calculate distances between all obstacles and all waypoints
            distances = cdist(obstacles_map_frame_array, relevant_waypoints)
            
            # Find the minimum distance for each obstacle to any waypoint in this lane
            min_distances = np.min(distances, axis=1)
            
            # If any obstacle is close enough to a waypoint, mark lane as blocked
            if np.any(min_distances < self.obstacle_proximity_threshold):
                blocked_indices = np.where(min_distances < self.obstacle_proximity_threshold)[0]
                if len(blocked_indices) > 0:
                    # Calculate the distance from car to closest obstacle in this lane
                    obstacle_idx = blocked_indices[0]
                    obstacle_x, obstacle_y = obstacles_map_frame[obstacle_idx]
                    dist_to_obstacle = np.sqrt((obstacle_x - car_x)**2 + (obstacle_y - car_y)**2)
                    
                    self.lane_status[lane_name]['blocked'] = True
                    self.lane_status[lane_name]['distance_to_obstacle'] = dist_to_obstacle
                    self.get_logger().info(f"Lane {lane_name} is blocked. Obstacle at distance: {dist_to_obstacle:.2f}m")
        
        # Check if current lane is blocked and switch if needed
        self.check_and_switch_lanes()

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def check_and_switch_lanes(self):
        """Check if current lane is blocked and switch to an unblocked lane if needed"""
        if not self.enable_lane_switching:
            return
            
        # Check if we're on cooldown for lane switching
        current_time = self.get_clock().now()
        time_since_last_switch = (current_time - self.last_lane_switch_time).nanoseconds / 1e9
        
        if time_since_last_switch < self.lane_switch_cooldown:
            return
            
        current_lane_status = self.lane_status[self.active_lane]
        
        if current_lane_status['blocked']:
            # Find an unblocked lane
            available_lanes = []
            for lane, status in self.lane_status.items():
                if not status['blocked'] and lane != self.active_lane:
                    available_lanes.append(lane)
            
            # If there are unblocked lanes, switch to the first one
            if available_lanes:
                new_lane = available_lanes[0]
                self.get_logger().info(f"Switching from {self.active_lane} to {new_lane} to avoid obstacle")
                self.set_active_lane(new_lane)
                self.last_lane_switch_time = current_time
            else:
                self.get_logger().warning("All lanes are blocked! Staying on current lane.")

    def publish_obstacle_markers(self, obstacles):
        """Publish markers for visualizing obstacles"""
        if not self.map_sim:
            return
            
        marker_array = MarkerArray()
        
        if self.current_pose is None:
            return
            
        # Get car position and orientation
        car_x = self.current_pose.position.x
        car_y = self.current_pose.position.y
        pose_dict = {'pose': self.current_pose}
        car_yaw = self.get_yaw_from_pose(self.current_pose.orientation)
        
        for i, (obs_x, obs_y) in enumerate(obstacles):
            # Transform obstacle from car frame to map frame
            map_x = car_x + obs_x * np.cos(car_yaw) - obs_y * np.sin(car_yaw)
            map_y = car_y + obs_x * np.sin(car_yaw) + obs_y * np.cos(car_yaw)
            
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "obstacles"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = map_x
            marker.pose.position.y = map_y
            marker.pose.position.z = 0.1  # Slightly above ground
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            
            marker_array.markers.append(marker)
        
        # If there are fewer markers this time, delete the old ones
        current_marker_count = len(marker_array.markers)
        previous_marker_count = len(self.obstacle_marker_array.markers)
        
        for i in range(current_marker_count, previous_marker_count):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "obstacles"
            marker.id = i
            marker.action = Marker.DELETE
            marker_array.markers.append(marker)
        
        self.obstacle_marker_array = marker_array
        self.obstacle_marker_publisher.publish(marker_array)

    def smooth_waypoints(self, lane_name):
        if not self.lanes[lane_name]:
            self.get_logger().warning(f"No waypoints to smooth for {lane_name}")
            return
            
        waypoints = np.array(self.lanes[lane_name])
        x = waypoints[:, 0]
        y = waypoints[:, 1]
        t = np.linspace(0, 1, len(x))

        # Skip if too few points
        if len(x) < 3:
            self.get_logger().warning(f"Not enough waypoints to smooth for {lane_name}")
            return

        # 2D B-Spline
        try:
            tck, u = splprep([x, y], s=8.0, k=2, per=True)  # k=3 for cubic spline
            t_smooth = np.linspace(0, 1, 1000)
            x_smooth, y_smooth = splev(t_smooth, tck)
            self.lanes[lane_name] = np.vstack((x_smooth, y_smooth)).T.tolist()
        except Exception as e:
            self.get_logger().error(f"Error smoothing waypoints for {lane_name}: {e}")

    def pose_callback(self, pose_msg):
        # Store current pose for obstacle detection
        pose = pose_msg.pose.pose

        # --- adaptive look‑ahead -----------------------------------------------
        κ = self.compute_path_curvature(pose)          # new helper
        self.adjust_lookahead_distance(κ)              # updates self.lookahead_distance
        # --- end: adaptive look‑ahead ------------------------------------

        
        # Find the current waypoint to track in the active lane
        current_waypoint = self.find_current_waypoint(pose)

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
            
            steering_angle = np.arctan(curvature * self.lookahead_distance * self.curvature_gain)

            # Publish drive message
            self.publish_drive_message(steering_angle)
            
            
    def compute_path_curvature(self, pose, sample_size=5):
        """
        Three‑point estimate of path curvature κ (1/m) around the car.
        """
        car_xy = np.array([pose.position.x, pose.position.y])

        # nearest waypoint index
        wp = np.array(self.lanes[self.active_lane]) 
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
        
        # Get waypoints from active lane
        waypoints = self.lanes[self.active_lane]
        if not waypoints:
            self.get_logger().error(f"No waypoints available for {self.active_lane}")
            return None

        closest_waypoint = None
        next_waypoint = None
        for i in range(len(waypoints) - 1):
            wp1 = waypoints[i]
            wp2 = waypoints[i + 1]
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
    
    def load_waypoints(self, csv_file_path, lane_name):
        try:
            with open(csv_file_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    try:
                        x, y = float(row[0]), float(row[1])
                        if len(row) >= 4 and float(row[3]) != 0:  # Assuming row[3] is a validity flag
                            self.lanes[lane_name].append((x, y))
                    except (ValueError, IndexError) as e:
                        self.get_logger().warning(f"Error parsing waypoint row: {e}")
            self.get_logger().info(f"Loaded {len(self.lanes[lane_name])} waypoints for {lane_name}")
        except FileNotFoundError:
            self.get_logger().error(f"Waypoint file not found: {csv_file_path}")
        except Exception as e:
            self.get_logger().error(f"Error loading waypoints for {lane_name}: {e}")

    def publish_waypoints_markers(self, current_waypoint=None):
        if not self.map_sim:
            return
            
        marker_array = self.marker_array
        marker_array.markers.clear()
        
        # Get all lanes for visualization with different colors
        lane_colors = {
            'lane1': (0.0, 1.0, 0.0),  # Green
            'lane2': (0.0, 0.0, 1.0),  # Blue
            'lane3': (1.0, 1.0, 0.0)   # Yellow
        }
        
        marker_id = 0
        
        # Add markers for all lanes
        for lane_name, waypoints in self.lanes.items():
            if not waypoints:
                continue
                
            color = lane_colors.get(lane_name, (0.5, 0.5, 0.5))  # Default to gray
            is_active = lane_name == self.active_lane
            is_blocked = self.lane_status[lane_name]['blocked']
            
            # If lane is blocked, show it in red
            if is_blocked:
                color = (1.0, 0.0, 0.0)  # Red for blocked lanes
            
            for i, waypoint in enumerate(waypoints):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = lane_name
                marker.id = marker_id
                marker_id += 1
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position = Point(x=waypoint[0], y=waypoint[1], z=0.0)
                marker.pose.orientation.w = 1.0
                
                # Make active lane markers bigger
                if is_active:
                    marker.scale.x = 0.1
                    marker.scale.y = 0.1
                    marker.scale.z = 0.1
                else:
                    marker.scale.x = 0.05
                    marker.scale.y = 0.05
                    marker.scale.z = 0.05
                
                marker.color.a = 1.0
                marker.color.r = color[0]
                marker.color.g = color[1]
                marker.color.b = color[2]
                
                marker_array.markers.append(marker)
        
        # Add current target waypoint if available
        if current_waypoint:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "current_target"
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = Point(x=current_waypoint[0], y=current_waypoint[1], z=0.0)
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
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
        speed = max_speed * (1 - (abs(steering_angle) / max_steering_angle))
        speed = max(min_speed, min(max_speed, speed))  # Ensure speed is within bounds
        
        # Slow down if obstacle is detected in current lane
        if self.lane_status[self.active_lane]['blocked']:
            obstacle_distance = self.lane_status[self.active_lane]['distance_to_obstacle']
            # Gradually slow down as we approach the obstacle
            slowdown_factor = min(1.0, obstacle_distance / self.obstacle_detection_distance)
            speed = speed * slowdown_factor
            
        drive_msg.drive.speed = speed
        self.current_speed = speed
        self.drive_publisher.publish(drive_msg)
        
    def set_active_lane(self, lane_name):
        """Switch to a different lane"""
        if lane_name in self.lanes:
            self.active_lane = lane_name
            self.get_logger().info(f"Switched to {lane_name}")
            # Update visualization
            # if self.map_sim:
                # self.publish_waypoints_markers()
        else:
            self.get_logger().error(f"Lane {lane_name} does not exist")

    # def get_yaw_from_pose(self, orientation):
    #     # Convert quaternion to yaw
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
    print("PurePursuit Initialized with Multiple Lanes and Obstacle Detection")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()