cmake_minimum_required(VERSION 3.8)
project(mppi)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(autoware_auto_planning_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

add_executable(mppi_controller_node 
  src/mppi_cuda.cpp
)

# Link dependencies:
ament_target_dependencies(mppi_controller_node
  rclcpp
  nav_msgs
  sensor_msgs
  geometry_msgs
  ackermann_msgs
  autoware_auto_planning_msgs
  tf2_ros
  tf2_geometry_msgs
)

# Install executable:
install(TARGETS mppi_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install headers:
install(DIRECTORY include/
  DESTINATION include/
)

target_include_directories(mppi_controller_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

install(PROGRAMS
  scripts/mppi_controller.py
  scripts/path_planner.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY resources DESTINATION share/${PROJECT_NAME}/)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
