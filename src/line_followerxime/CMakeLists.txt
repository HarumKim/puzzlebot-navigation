cmake_minimum_required(VERSION 3.8)
project(line_followerxime)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ===============================
# FIND REQUIRED PACKAGES
# ===============================

# C++ Core
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)

# Python Support
find_package(ament_cmake_python REQUIRED)

# ===============================
# C++ EXECUTABLES
# ===============================

add_executable(camera src/camera.cpp)
ament_target_dependencies(camera
  rclcpp
  sensor_msgs
  image_transport
  OpenCV
  cv_bridge
)

# ===============================
# INCLUDE DIRECTORIES
# ===============================
target_include_directories(camera PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(camera ${OpenCV_LIBRARIES})

# ===============================
# INSTALLATION
# ===============================

# Install C++ node
install(TARGETS camera
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python package (scripts)
ament_python_install_package(${PROJECT_NAME})

# Install Python nodes
install(PROGRAMS
  line_followerxime/line_follower.py
  line_followerxime/light_detector.py
  line_followerxime/testmodel.py
  line_followerxime/hsv_calibra.py
  line_followerxime/image_capturer.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# ===============================
# FINALIZE
# ===============================

ament_package()
