cmake_minimum_required(VERSION 3.8)
project(mppi LANGUAGES CXX CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# CUDA setup
find_package(CUDA REQUIRED)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

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
find_package(Eigen3 REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# CUDA executable with .cu files
add_executable(mppi_controller_node 
  src/mppi_cuda.cpp
  src/mppi_kernel.cu
)

# Set CUDA flags
target_compile_options(mppi_controller_node PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    --use_fast_math
    -Wno-pedantic
  >
)

# Include directories
target_include_directories(mppi_controller_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CUDA_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# Link dependencies
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

# Link CUDA libraries
target_link_libraries(mppi_controller_node
  ${CUDA_LIBRARIES}
  cudart
  curand
)

# Install executable
install(TARGETS mppi_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install headers
install(DIRECTORY include/
  DESTINATION include/
)

install(PROGRAMS
  scripts/mppi_controller.py
  scripts/path_planner.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY resources DESTINATION share/${PROJECT_NAME}/)

# Shared CUDA kernels library
add_library(mppi_kernels
  src/mppi_kernel.cu  # Implementation only
)
target_include_directories(mppi_kernels PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(mppi_kernels
  ${CUDA_LIBRARIES}
  cudart
  curand
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  
  # dynamics test
  ament_add_gtest(test_dynamics test/test_dynamics.cpp)
  target_include_directories(test_dynamics PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
  )
  target_link_libraries(test_dynamics
    ${CUDA_LIBRARIES}
  )
  
  # kernel tests
  add_library(test_kernel_device 
    test/test_kernel_device.cu  # Only test wrappers
  )
  target_link_libraries(test_kernel_device
    mppi_kernels
    ${CUDA_LIBRARIES}
    cudart
    curand
  )
  
  ament_add_gtest(test_kernel test/test_kernel.cpp)
  target_link_libraries(test_kernel
    test_kernel_device
    ${CUDA_LIBRARIES}
  )
  
  # integration tests
  # ament_add_gtest(test_mppi_controller
  #   test/test_mppi_controller.cpp
  # )
  # ament_target_dependencies(test_mppi_controller
  #   rclcpp
  # )
  # target_include_directories(test_mppi_controller PUBLIC
  #   ${CMAKE_CURRENT_SOURCE_DIR}/include
  # )
  
endif()

ament_package()