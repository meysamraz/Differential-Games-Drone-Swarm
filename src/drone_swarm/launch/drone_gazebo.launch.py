"""
Launch 4 quadrotors: 3 formation drones (scattered spawn) + 1 evader.

Formation drones converge from scattered positions to triangle using Nash equilibrium.
Evader idles at (6, 6) until /formation_cmd = 'pursuit' is sent from formation_ui.

Demo sequence:
  1. Drones converge scattered → triangle (Nash formation)
  2. formation_ui: switch triangle → line → v_shape → diamond
  3. formation_ui: START PURSUIT → drone1 chases evader, 2+3 trail in V
  4. formation_ui: RETURN TO FORMATION → back to last shape
  5. wind_controller: crank to HURRICANE to stress-test recovery

Topics:
  /droneN/odom, /droneN/cmd_vel, /droneN/cmd_force   (N = 1,2,3)
  /evader/odom, /evader/cmd_vel, /evader/cmd_force
  /formation_cmd   ← formation_ui
  /wind_scale      ← wind_controller
"""

import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


# Formation drones — scattered spawn so convergence is visible
DRONES = [
    {'ns': 'drone1', 'x': -4.00, 'y': -2.00, 'z': 0.15},
    {'ns': 'drone2', 'x':  5.00, 'y': -3.00, 'z': 0.15},
    {'ns': 'drone3', 'x':  0.50, 'y':  6.00, 'z': 0.15},
]

# Evader — spawns near the formation centroid so surround/pursuit engages quickly
EVADER = {'ns': 'evader', 'x': 2.00, 'y': 1.00, 'z': 0.15}

HOVER_ALT      = 1.0   # m  – all drones auto-hover here after spawning
SPAWN_START    = 3.0   # s  – first spawn after Gazebo is ready
SPAWN_INTERVAL = 1.2   # s  – stagger between spawns (avoids Gazebo race)
CTRL_DELAY     = 8.0   # s  – controllers start after all drones are spawned

# Per-drone Itô noise σ and actuation delay (matching simulation heterogeneity)
# Leader is most stable, Scout is noisiest
NOISE_SIGMA  = [0.08, 0.15, 0.22]   # Itô noise intensity per drone
DELAY_STEPS  = [3,    3,    3   ]   # actuation delay steps at 100 Hz (= 30 ms)


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

        # Flight controller with per-drone noise and delay
        actions.append(TimerAction(
            period=CTRL_DELAY,
            actions=[Node(
                package='drone_swarm',
                executable='drone_controller',
                name='drone_controller',
                namespace=ns,
                parameters=[{
                    'hover_altitude': HOVER_ALT,
                    'noise_sigma':    NOISE_SIGMA[i],
                    'delay_steps':    DELAY_STEPS[i],
                }],
                output='screen',
            )],
        ))

    # ------------------------------------------------ Evader drone (4th quadrotor)
    evader_urdf = xacro.process_file(
        xacro_path, mappings={'drone_ns': EVADER['ns']}
    ).toxml()
    actions.append(Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace=EVADER['ns'],
        name='robot_state_publisher',
        parameters=[{'robot_description': evader_urdf, 'use_sim_time': True}],
        output='screen',
    ))
    actions.append(TimerAction(
        period=SPAWN_START + len(DRONES) * SPAWN_INTERVAL,
        actions=[Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            name='spawn_evader',
            arguments=[
                '-entity', EVADER['ns'],
                '-topic',  f'/{EVADER["ns"]}/robot_description',
                '-x', str(EVADER['x']),
                '-y', str(EVADER['y']),
                '-z', str(EVADER['z']),
            ],
            output='screen',
        )],
    ))
    # Evader flight controller (altitude hold only — evader_controller handles XY)
    actions.append(TimerAction(
        period=CTRL_DELAY,
        actions=[Node(
            package='drone_swarm',
            executable='drone_controller',
            name='drone_controller',
            namespace=EVADER['ns'],
            parameters=[{
                'hover_altitude': HOVER_ALT,
                'noise_sigma':    0.05,
                'delay_steps':    2,
            }],
            output='screen',
        )],
    ))
    # Evader XY behaviour node
    actions.append(TimerAction(
        period=CTRL_DELAY + 4.0,
        actions=[Node(
            package='drone_swarm',
            executable='evader_controller',
            name='evader_controller',
            output='screen',
        )],
    ))

    # Wind node — starts with flight controllers, blows throughout
    actions.append(TimerAction(
        period=CTRL_DELAY,
        actions=[Node(
            package='drone_swarm',
            executable='wind_node',
            name='wind_node',
            output='screen',
        )],
    ))

    # Formation controller — starts after flight controllers have stabilised
    actions.append(TimerAction(
        period=CTRL_DELAY + 4.0,
        actions=[Node(
            package='drone_swarm',
            executable='formation_controller',
            name='formation_controller',
            output='screen',
        )],
    ))

    return LaunchDescription(actions)
