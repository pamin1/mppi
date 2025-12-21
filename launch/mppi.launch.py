#!/usr/bin/env python3
"""
Launch file for F1TENTH Racing with MPPI Controller

Launches:
1. F1TENTH Gym ROS simulator
2. Global trajectory publisher (path planner)
3. MPPI controller

Usage:
    ros2 launch mppi mppi_racing_launch.py

    # With custom map:
    ros2 launch mppi mppi_racing_launch.py map:=Monza_map

    # With custom trajectory:
    ros2 launch mppi mppi_racing_launch.py trajectory:=Monza_map_optimized.csv
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Package directories
    mppi_pkg = get_package_share_directory("mppi")
    f1tenth_gym_pkg = get_package_share_directory("f1tenth_gym_ros")

    # Launch arguments
    map_name_arg = DeclareLaunchArgument(
        "map",
        default_value="Spielberg_map",
        description="Name of the map (without extension)",
    )

    trajectory_file_arg = DeclareLaunchArgument(
        "trajectory",
        default_value="Spielberg_map_optimized.csv",
        description="Name of trajectory CSV file",
    )

    use_rviz_arg = DeclareLaunchArgument(
        "rviz", default_value="true", description="Launch RViz for visualization"
    )

    control_freq_arg = DeclareLaunchArgument(
        "control_freq", default_value="50", description="MPPI control frequency (Hz)"
    )

    # Get launch configurations
    map_name = LaunchConfiguration("map")
    trajectory_file = LaunchConfiguration("trajectory")
    use_rviz = LaunchConfiguration("rviz")
    control_freq = LaunchConfiguration("control_freq")

    # ==========================================================================
    # 2. GLOBAL TRAJECTORY PUBLISHER (PATH PLANNER)
    # ==========================================================================

    # Build trajectory file path
    trajectory_path = PathJoinSubstitution(
        [FindPackageShare("mppi"), "resources", trajectory_file]
    )

    trajectory_publisher = Node(
        package="mppi",
        executable="path_planner.py",
        name="path_planner",
        output="screen",
    )

    # ==========================================================================
    # 3. MPPI CONTROLLER
    # ==========================================================================

    # MPPI parameters file
    mppi_params_file = os.path.join(
        get_package_share_directory("mppi"), "config", "cost_weights.yaml"
    )

    mppi_controller = Node(
        package="mppi",
        executable="mppi_controller.py",
        name="mppi_controller",
        output="screen",
        parameters=[
            mppi_params_file,
        ],
    )

    # ==========================================================================
    # LAUNCH DESCRIPTION
    # ==========================================================================

    return LaunchDescription(
        [
            # Arguments
            map_name_arg,
            trajectory_file_arg,
            use_rviz_arg,
            control_freq_arg,
            # Nodes
            trajectory_publisher,  # Path planner
            mppi_controller,  # MPPI controller
        ]
    )
