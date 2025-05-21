cmake_minimum_required(VERSION 3.8)
project(image_viewer_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 🧩 Dependencias
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# 🚀 Ejecutable: image_publisher
#add_executable(image_publisher src/image_publisher.cpp)
#ament_target_dependencies(image_publisher
#  rclcpp
#  sensor_msgs
#  image_transport
#  cv_bridge
#)
#target_include_directories(image_publisher PUBLIC ${OpenCV_INCLUDE_DIRS})
#target_link_libraries(image_publisher ${OpenCV_LIBRARIES})

# 🚀 Ejecutable: image_subscriber
add_executable(image_subscriber src/image_subscriber.cpp)
ament_target_dependencies(image_subscriber
  rclcpp
  sensor_msgs
  image_transport
  cv_bridge
)
target_include_directories(image_subscriber PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(image_subscriber ${OpenCV_LIBRARIES})

# 🗃️ Instalación
install(TARGETS
  #image_publisher
  image_subscriber
  DESTINATION lib/${PROJECT_NAME}
)

# ✅ Lint (opcional)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
