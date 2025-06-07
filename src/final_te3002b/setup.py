from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'final_te3002b'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.py'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.[yma]*'))),
        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*.rviz'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mateo Sanchez',
    maintainer_email='neronf123@gmail.com',
    description='Line Follower Algorithm',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'line_follower = final_te3002b.line_follower:main',
            'traffic_light = final_te3002b.traffic_light:main',
            'traffic_signs = final_te3002b.traffic_signs:main',
            'state_machine = final_te3002b.state_machine:main',
        ],
    },
)
rm