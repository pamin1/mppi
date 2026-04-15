#!/usr/bin/env python3
"""
batch_launch.py — Parameterized ROS2 launch file for a single batch track run.

Launches the full F1TENTH + MPPI stack headlessly (no RViz).  All
track-specific settings and batch parameters are exposed as launch arguments
so that batch_runner.py can drive each run without modifying any source files.

Arguments
---------
trajectory        : CSV stem in share/mppi/resources/ (no .csv), e.g. track_01_optimized
sim_config        : absolute path to a per-track sim.yaml written by batch_runner
                    (contains map_path, sx, sy, stheta, kb_teleop=False)
target_laps       : stop after this many laps (int)
timeout_seconds   : hard time-limit for the run (float, seconds)
collision_threshold_m          : LiDAR range below which a collision is flagged (float)
collision_consecutive_readings : number of consecutive readings required (int)
min_lap_distance               : minimum distance before finish line is armed (float)
rerun_output      : absolute path for the .rrd recording file

Usage (invoked by batch_runner.py):
    ros2 launch mppi batch_launch.py \\
        trajectory:=track_01_optimized \\
        sim_config:=/abs/path/to/sim_track_01.yaml \\
        target_laps:=3 \\
        timeout_seconds:=120.0
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


# ── Argument declarations ─────────────────────────────────────────────────────

def _declare_args():
    f1tenth_share = get_package_share_directory("f1tenth_gym_ros")
    default_sim   = os.path.join(f1tenth_share, "config", "sim.yaml")

    return [
        DeclareLaunchArgument("trajectory", default_value="Spielberg_map_optimized"),
        DeclareLaunchArgument("sim_config", default_value=default_sim,
                              description="Absolute path to per-track sim.yaml"),
        DeclareLaunchArgument("target_laps",                    default_value="3"),
        DeclareLaunchArgument("timeout_seconds",                default_value="120.0"),
        DeclareLaunchArgument("collision_threshold_m",          default_value="0.12"),
        DeclareLaunchArgument("collision_consecutive_readings", default_value="3"),
        DeclareLaunchArgument("min_lap_distance",               default_value="30.0"),
        DeclareLaunchArgument("rerun_output",                   default_value="",
                              description="Absolute path for the .rrd recording file"),
        DeclareLaunchArgument("log_output",                     default_value="",
                              description="Absolute path for the run_log.csv written by monitor_node"),
        DeclareLaunchArgument("metrics_output_path",            default_value="",
                              description="Absolute path for the metrics.json written by monitor_node"),
    ]


# ── Node factory (runs after substitutions are resolved) ─────────────────────

def _launch_setup(context, *args, **kwargs):
    """
    OpaqueFunction callback: resolves all LaunchConfiguration values to strings
    and instantiates the node list.  Using OpaqueFunction lets us read the
    per-track sim.yaml at launch time to extract the map_path for map_server.
    """
    sim_config_path = LaunchConfiguration("sim_config").perform(context)
    trajectory      = LaunchConfiguration("trajectory").perform(context)
    target_laps     = LaunchConfiguration("target_laps").perform(context)
    timeout_secs    = LaunchConfiguration("timeout_seconds").perform(context)
    coll_thresh     = LaunchConfiguration("collision_threshold_m").perform(context)
    coll_n          = LaunchConfiguration("collision_consecutive_readings").perform(context)
    min_lap         = LaunchConfiguration("min_lap_distance").perform(context)
    rerun_output        = LaunchConfiguration("rerun_output").perform(context)
    log_output          = LaunchConfiguration("log_output").perform(context)
    metrics_output_path = LaunchConfiguration("metrics_output_path").perform(context)

    # Read sim config to get map_path (needed by map_server)
    with open(sim_config_path) as f:
        sim_cfg = yaml.safe_load(f)
    map_path = sim_cfg["bridge"]["ros__parameters"]["map_path"]
    map_yaml = map_path + ".yaml"

    f1tenth_share = get_package_share_directory("f1tenth_gym_ros")
    mppi_share    = get_package_share_directory("mppi")

    # ── 1. F1TENTH gym bridge (headless — kb_teleop set to False in sim_config)
    bridge_node = Node(
        package="f1tenth_gym_ros",
        executable="gym_bridge",
        name="bridge",
        parameters=[sim_config_path],
        output="log",
    )

    # ── 2. Map server
    map_server_node = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        parameters=[{
            "yaml_filename": map_yaml,
            "topic":         "map",
            "frame_id":      "map",
            "use_sim_time":  True,
        }],
        output="log",
    )

    # ── 3. Nav2 lifecycle manager (activates map_server)
    lifecycle_node = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="log",
        parameters=[{
            "use_sim_time": True,
            "autostart":    True,
            "node_names":   ["map_server"],
        }],
    )

    # ── 4. Robot state publisher (ego only — single-agent batch runs)
    ego_robot_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="ego_robot_state_publisher",
        parameters=[{
            "robot_description": Command([
                "xacro ",
                os.path.join(f1tenth_share, "launch", "ego_racecar.xacro"),
            ])
        }],
        remappings=[("/robot_description", "ego_robot_description")],
    )

    # ── 5. Path planner (global trajectory publisher)
    # NOTE: path_planner.py currently hard-codes 'Spielberg_map_optimized.csv'.
    # We pass 'trajectory_file' as a parameter so that once path_planner.py is
    # updated to read it the batch framework will work for all tracks without
    # further changes.
    path_planner_node = Node(
        package="mppi",
        executable="path_planner.py",
        name="path_planner",
        output="log",
        parameters=[{"trajectory_file": trajectory}],
    )

    # ── 6. MPPI controller (fixed config across all tracks)
    mppi_params = os.path.join(mppi_share, "config", "cost_weights.yaml")
    mppi_controller_node = Node(
        package="mppi",
        executable="mppi_controller_node",
        name="mppi_controller_node",
        output="log",
        parameters=[mppi_params],
    )

    # ── 7. Monitor node (batch-parameterised)
    monitor_node = Node(
        package="mppi",
        executable="monitor_node.py",
        name="monitor_node",
        output="screen",
        parameters=[{
            "target_laps":                    int(target_laps),
            "timeout_seconds":                float(timeout_secs),
            "collision_threshold_m":          float(coll_thresh),
            "collision_consecutive_readings": int(coll_n),
            "min_lap_distance":               float(min_lap),
            "rerun_output":                   rerun_output,
            "log_output":                     log_output,
            "metrics_output_path":            metrics_output_path,
        }],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', os.path.join(get_package_share_directory('f1tenth_gym_ros'), 'launch', 'gym_bridge.rviz')],
        output="log"
    )

    # When monitor_node exits (after publishing its final status), shut down
    # the entire launch — bridge, planner, controller, rviz all get SIGINT.
    teardown_on_monitor_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=monitor_node,
            on_exit=[EmitEvent(event=Shutdown())],
        )
    )

    return [
        bridge_node,
        map_server_node,
        lifecycle_node,
        ego_robot_publisher,
        path_planner_node,
        mppi_controller_node,
        monitor_node,
        rviz_node,
        teardown_on_monitor_exit,
    ]


# ── Entry point ───────────────────────────────────────────────────────────────

def generate_launch_description():
    return LaunchDescription(
        _declare_args() + [OpaqueFunction(function=_launch_setup)]
    )
