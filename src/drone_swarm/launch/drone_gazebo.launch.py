"""
Launch Gazebo Classic with one quadrotor drone + flight controller.

Usage:
  ros2 launch drone_swarm drone_gazebo.launch.py

Then in another terminal:
  ros2 run teleop_twist_keyboard teleop_twist_keyboard
    t / b      → up / down
    i / ,      → forward / backward
    j / l      → yaw left / yaw right
    J / L      → strafe left / right  (capital – holonomic mode)
    k          → full stop / hover hold
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('drone_swarm')
    urdf_path = os.path.join(pkg, 'urdf', 'quadrotor.urdf')
    world_path = os.path.join(pkg, 'worlds', 'drone_world.world')

    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    # 1. Gazebo Classic with ROS2 bridge plugins
    gazebo = ExecuteProcess(
        cmd=[
            'gazebo', '--verbose', world_path,
            '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so',
        ],
        output='screen',
    )

    # 2. Robot state publisher (publishes /tf from URDF joints)
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': robot_description,
                     'use_sim_time': True}],
        output='screen',
    )

    # 3. Spawn drone (delayed so Gazebo is up)
    spawn = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                name='spawn_drone',
                arguments=[
                    '-entity', 'quadrotor',
                    '-topic', '/robot_description',
                    '-x', '0.0',
                    '-y', '0.0',
                    '-z', '0.15',   # spawn just above ground
                ],
                output='screen',
            )
        ],
    )

    # 4. Flight controller (delayed until drone is spawned)
    controller = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='drone_swarm',
                executable='drone_controller',
                name='drone_controller',
                output='screen',
            )
        ],
    )

    return LaunchDescription([gazebo, rsp, spawn, controller])
