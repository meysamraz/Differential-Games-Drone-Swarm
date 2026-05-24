from setuptools import find_packages, setup
import glob
import os

package_name = 'drone_swarm'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/urdf',
            [f for f in glob.glob('urdf/*') if os.path.isfile(f)]),
        ('share/' + package_name + '/urdf/parts',
            glob.glob('urdf/parts/*')),
        ('share/' + package_name + '/launch', glob.glob('launch/*')),
        ('share/' + package_name + '/worlds', glob.glob('worlds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='meisam',
    maintainer_email='mr.raz2002@gmail.com',
    description='Drone swarm with differential games – single drone step',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'drone_controller      = drone_swarm.drone_controller:main',
            'drone_teleop          = drone_swarm.drone_teleop:main',
            'formation_controller  = drone_swarm.formation_controller:main',
            'wind_node             = drone_swarm.wind_node:main',
            'wind_controller       = drone_swarm.wind_controller:main',
            'cost_plotter          = drone_swarm.cost_plotter:main',
            'formation_ui          = drone_swarm.formation_ui:main',
            'evader_controller     = drone_swarm.evader_controller:main',
        ],
    },
)
