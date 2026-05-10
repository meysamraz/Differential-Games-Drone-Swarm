"""
Launch 3 quadrotors in Gazebo at equilateral-triangle positions.

Each drone is fully independent – separate namespace, RSP, Gazebo entity,
force plugin, odometry, and controller node.

Namespace layout
----------------
/drone1   →  x= 0.00, y= 0.00   (origin)
/drone2   →  x= 2.00, y= 0.00   (right)
/drone3   →  x= 1.00, y= 1.73   (top – equilateral triangle, side 2 m)

Topics per drone (example for /drone1)
---------------------------------------
  /drone1/odom        ← libgazebo_ros_p3d  (ground-truth odometry)
  /drone1/cmd_force   ← drone_controller   → libgazebo_ros_force
  /drone1/cmd_vel     ← user / formation commander

To teleop a single drone
------------------------
  ros2 run drone_swarm drone_teleop --ros-args -r /cmd_vel:=/drone1/cmd_vel
"""

import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


# --- Equilateral triangle with 2 m side length, z = 0.15 m (just above ground) ---
DRONES = [
    {'ns': 'drone1', 'x':  0.00, 'y':  0.00, 'z': 0.15},
    {'ns': 'drone2', 'x':  2.00, 'y':  0.00, 'z': 0.15},
    {'ns': 'drone3', 'x':  1.00, 'y':  1.73, 'z': 0.15},
]

HOVER_ALT      = 1.0   # m  – all drones auto-hover here after spawning
SPAWN_START    = 3.0   # s  – first spawn after Gazebo is ready
SPAWN_INTERVAL = 1.2   # s  – stagger between spawns (avoids Gazebo race)
CTRL_DELAY     = 8.0   # s  – controllers start after all drones are spawned


def generate_launch_description():
    pkg        = get_package_share_directory('drone_swarm')
    xacro_path = os.path.join(pkg, 'urdf', 'quadrotor.urdf.xacro')
    world_path = os.path.join(pkg, 'worlds', 'drone_world.world')

    actions = []

    # ------------------------------------------------------------------ Gazebo
    actions.append(ExecuteProcess(
        cmd=[
            'gazebo', '--verbose', world_path,
            '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so',
        ],
        output='screen',
    ))

    # ----------------------------------------------- Per-drone RSP + spawn + ctrl
    for i, drone in enumerate(DRONES):
        ns = drone['ns']
        x, y, z = drone['x'], drone['y'], drone['z']

        # xacro processes the template and injects the namespace for this drone
        drone_urdf = xacro.process_file(
            xacro_path,
            mappings={'drone_ns': ns}
        ).toxml()

        # Robot state publisher (publishes /{ns}/robot_description for the spawner)
        actions.append(Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            namespace=ns,
            name='robot_state_publisher',
            parameters=[{
                'robot_description': drone_urdf,
                'use_sim_time': True,
            }],
            output='screen',
        ))

        # Spawn entity – staggered so Gazebo doesn't get overloaded
        spawn_time = SPAWN_START + i * SPAWN_INTERVAL
        actions.append(TimerAction(
            period=spawn_time,
            actions=[Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                name=f'spawn_{ns}',
                arguments=[
                    '-entity', ns,
                    '-topic',  f'/{ns}/robot_description',
                    '-x', str(x),
                    '-y', str(y),
                    '-z', str(z),
                ],
                output='screen',
            )],
        ))

        # Flight controller – starts after all 3 drones are in the world
        actions.append(TimerAction(
            period=CTRL_DELAY,
            actions=[Node(
                package='drone_swarm',
                executable='drone_controller',
                name='drone_controller',
                namespace=ns,
                parameters=[{'hover_altitude': HOVER_ALT}],
                output='screen',
            )],
        ))

    return LaunchDescription(actions)
