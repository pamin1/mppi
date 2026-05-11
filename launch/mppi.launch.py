#!/usr/bin/env python3
"""
Launch file for F1TENTH Racing with MPPI Controller

Usage:
    ros2 launch mppi mppi_racing_launch.py
    ros2 launch mppi mppi_racing_launch.py map:=wide_oval
"""

import os
from datetime import datetime
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ld = LaunchDescription()

    timestamp = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    bag_directory = os.path.join('bags', f'record_{timestamp}')
    
    mppi_pkg = get_package_share_directory("mppi")
    f1tenth_pkg = get_package_share_directory("f1tenth_gym_ros")

    params_file = os.path.join(mppi_pkg, "config", "sim.yaml")

    # --- Arguments ---
    map_arg = DeclareLaunchArgument(
        "map", default_value="Spielberg_map",
        description="Map name (without extension)",
    )

    bagging = DeclareLaunchArgument(
        "bag", default_value="False",
        description="Write ros bags"
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
        parameters=[{
            "trajectory_path": trajectory_path,
            "control_dt": 1 / 60.0
            }],
    )

    local_map = Node(
        package="mppi",
        executable="local_map.py",
        name="local_map",
        output="screen",
        parameters=[params_file],
    )

    mppi_controller = Node(
        package="mppi",
        executable="mppi_controller_node",
        name="mppi_controller",
        output="screen",
        parameters=[params_file],
    )

    lap_logger = Node(
        package="mppi",
        executable="lap_logger.py",
        name="bridge",
        output="screen",
        parameters=[params_file]
    )

    f1tenth_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(f1tenth_pkg, "launch", "gym_bridge_launch.py")
        ),
        launch_arguments={"params_file": params_file}.items(),
    )

    bag = ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-a', '-o', bag_directory],
            output='screen',
            condition=IfCondition(LaunchConfiguration("bag"))
        )

    ld.add_action(map_arg)
    ld.add_action(bagging)
    ld.add_action(path_planner)
    ld.add_action(local_map)
    ld.add_action(mppi_controller)
    ld.add_action(lap_logger)
    ld.add_action(f1tenth_sim)
    ld.add_action(bag)

    return ld
