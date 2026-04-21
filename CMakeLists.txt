cmake_minimum_required(VERSION 3.8)
project(mppi LANGUAGES CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-10)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)
enable_language(CUDA)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_costmap_2d)
find_package(tf2_ros REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(autoware_auto_planning_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

ament_python_install_package(${PROJECT_NAME})
add_library(mppi_kernels
  src/mppi_kernel.cu
)

target_include_directories(mppi_kernels PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(mppi_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    --use_fast_math
    -ccbin=/usr/bin/g++-10
  >
)

target_link_libraries(mppi_kernels
  CUDA::cudart
  CUDA::curand
  Thrust
)

add_executable(mppi_controller_node 
  src/mppi_cuda.cpp
)

target_include_directories(mppi_controller_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)

ament_target_dependencies(mppi_controller_node
  rclcpp
  nav_msgs
  sensor_msgs
  geometry_msgs
  ackermann_msgs
  autoware_auto_planning_msgs
  tf2_ros
  tf2_geometry_msgs
  visualization_msgs
)

target_link_libraries(mppi_controller_node
  mppi_kernels
  CUDA::cudart
  CUDA::curand
)

install(TARGETS mppi_controller_node mppi_kernels
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

install(PROGRAMS
  scripts/path_planner.py
  scripts/local_map.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY resources DESTINATION share/${PROJECT_NAME}/)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  
  ament_add_gtest(test_dynamics test/test_dynamics.cpp)
  target_include_directories(test_dynamics PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  
  add_library(test_kernel_device 
    test/test_kernel_device.cu
  )
  target_compile_options(test_kernel_device PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      -ccbin=/usr/bin/g++-10
    >
  )
  target_link_libraries(test_kernel_device
    mppi_kernels
    CUDA::cudart
    CUDA::curand
    Thrust
  )
  
  ament_add_gtest(test_kernel test/test_kernel.cpp)
  target_link_libraries(test_kernel
    test_kernel_device
  )
endif()

ament_package()