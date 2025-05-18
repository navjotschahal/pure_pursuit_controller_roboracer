import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
   config = os.path.join(
      get_package_share_directory('pure_pursuit'),
      'config',
      'pure_pursuit.params.yaml'
      )

   return LaunchDescription([
      Node(
         package='pure_pursuit',
         executable='pure_pursuit_node_d.py',
         namespace='',
         name='pure_pursuit_node_d',
         parameters=[config]
      )
   ])