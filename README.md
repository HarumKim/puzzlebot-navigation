# 🤖 Puzzlebot Line Follower with Traffic Sign & Light Recognition (ROS 2)

This project implements an **autonomous navigation system** for a differential-drive Puzzlebot robot using **ROS 2 Humble**. The robot follows a predefined track using visual line detection and reacts autonomously to **traffic signs** and **traffic lights**, combining computer vision, control theory, and state-based decision making.

---

## 🎯 Project Overview

The system is built using a **modular ROS 2 architecture**, where multiple nodes handle perception, control, and decision-making independently. A central **state machine** coordinates the robot’s behavior based on visual inputs and detected environmental cues.

The robot is capable of:
- Following a black centerline using a camera
- Detecting traffic signs and traffic lights using YOLO-based vision
- Adjusting speed and behavior based on detected signals
- Providing auditory feedback through a buzzer

---

## 🧠 System Architecture

The final implementation is contained in the `final_te3002b` package. Other folders in the repository correspond to previous exercises or partial developments.

Main nodes in the system:

- **camera**  
  Captures raw images and publishes a compressed video stream to reduce latency.

- **line_follower.py**  
  Implements visual line following using PID control and publishes velocity commands.

- **traffic_signs.py**  
  Detects road signs using a trained YOLO model and publishes the detected sign.

- **traffic_light.py**  
  Detects traffic light color and publishes the result when the robot reaches an intersection.

- **state_machine.py**  
  Acts as the master node. It integrates all perception inputs and decides the robot’s behavior.

- **odometry.py**  
  Computes robot motion estimates for navigation and future extensions.

All nodes communicate through ROS 2 topics, allowing clear separation of responsibilities and easier debugging.

---

## 🔧 Requirements

- ROS 2 **Humble**
- Python 3
- NVIDIA Jetson Nano (for onboard execution)
- Puzzlebot (Manchester Robotics)
- Raspberry Pi camera (fish-eye lens recommended)
- Passive buzzer (GPIO-controlled)
- OpenCV, NumPy, YOLO dependencies

---

## 🚀 How to Run the Project

### 1. Build the Workspace

From your ROS 2 workspace root:

```bash
colcon build
source install/setup.bash
```

### 2. Launch the System

The entire pipeline is launched using a single launch file:

```bash
ros2 launch final_te3002b track.launch.py
```

This launch file automatically starts the main nodes of the system. Once launched, the robot will begin autonomous navigation on the track.

--
## 🎥 Demonstration

Full Autonomous Navigation Demo
Shows line following, traffic sign reactions, and traffic light handling.
🔗 https://youtu.be/AYWiGJdpJkY
