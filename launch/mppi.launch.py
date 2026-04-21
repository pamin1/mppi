#!/usr/bin/env python3
"""
Launch file for F1TENTH Racing with MPPI Controller

Usage:
    ros2 launch mppi mppi_racing_launch.py
    ros2 launch mppi mppi_racing_launch.py map:=wide_oval
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    mppi_pkg = get_package_share_directory("mppi")
    f1tenth_pkg = get_package_share_directory("f1tenth_gym_ros")

    params_file = os.path.join(mppi_pkg, "config", "sim.yaml")

    # --- Arguments ---
    map_arg = DeclareLaunchArgument(
        "map", default_value="Spielberg_map",
        description="Map name (without extension)",
    )

    # --- Derived substitutions ---
    map_name = LaunchConfiguration("map")
    trajectory_file = PythonExpression(["'", map_name, "_optimized.csv'"])
    trajectory_path = PathJoinSubstitution(
        [FindPackageShare("mppi"), "resources", trajectory_file]
    )

    # --- Nodes ---
    path_planner = Node(
        package="mppi",
        executable="path_planner.py",
        name="path_planner",
        output="screen",
        parameters=[{"trajectory_path": trajectory_path}],
    )

    local_map = Node(
        package="mppi",
        executable="local_map.py",
        name="local_map",
        output="screen",
    )

    mppi_controller = Node(
        package="mppi",
        executable="mppi_controller_node",
        name="mppi_controller",
        output="screen",
        parameters=[params_file],
    )

    f1tenth_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(f1tenth_pkg, "launch", "gym_bridge_launch.py")
        ),
        launch_arguments={"params_file": params_file}.items(),
    )

    return LaunchDescription([
        map_arg,
        path_planner,
        local_map,
        mppi_controller,
        f1tenth_sim,
    ])