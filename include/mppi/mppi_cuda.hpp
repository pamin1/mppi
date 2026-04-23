#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <autoware_auto_planning_msgs/msg/trajectory.hpp>
#include <chrono>
#include <mppi/gap_detector.hpp>
#include <mppi/kernel_launch.hpp>
#include <mppi/mppi_util.hpp>
#include <mppi/util.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <random>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

class MPPI_Controller : public rclcpp::Node
{
  public:
    MPPI_Controller();
    ~MPPI_Controller();

    void loadParameters();
    void updateState(const nav_msgs::msg::Odometry::SharedPtr odom);
    void updateTraj(const autoware_auto_planning_msgs::msg::Trajectory::SharedPtr traj);
    void updateMap(const nav_msgs::msg::OccupancyGrid::SharedPtr map);
    void updateScan(const sensor_msgs::msg::LaserScan::SharedPtr scan);

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void trajectoryCallback(const autoware_auto_planning_msgs::msg::Trajectory::SharedPtr msg);
    void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg);

    void updateControl();
    void publishModes(const std::vector<Gaussian> &modes, const rclcpp::Time &stamp, int best_mode);
    void publishBestPath(int best_mode);

  private:
    static constexpr int MAX_MODES = 4;

    // controller set up
    std::string mapFrame = "map";
    std::string baseFrame = "ego_racecar/base_link";
    double currentSteeringAngle = 0;

    MPPIConfig config;
    int samples;
    int controlFrequency;
    float dt;
    int horizon; // timesteps, not time
    double temperature;
    double alpha;

    // MPPI givens
    bool controlSeqInitialized = false;
    std::vector<ControlInput> nominalControlSequence;
    std::vector<VehicleState> trajectory;
    float sigmaAcceleration, sigmaSteering; // control noise for sampling

    // visualization
    bool enableViz;
    int topKPaths;
    float vizLineWidth;
    float vizPathAlpha;

    // GPU arrays
    int block, grid;

    ControlInput *d_optimalControls;
    ControlInput *d_optimal_per_mode[MAX_MODES];

    ControlInput *d_nominalControls;
    VehicleState *d_refTraj;
    VehicleState *d_currState;

    CostWeights *d_weights;
    VehicleParams *d_params;

    curandState *d_rngStates;

    ControlInput *d_controls;
    ControlInput *d_controls_per_mode[MAX_MODES];
    double *d_costs;
    double *d_costs_per_mode[MAX_MODES];

    int8_t *d_costmap_data;
    CostmapInfo *d_costmap_info;

    // create multiple cuda streams
    cudaStream_t streams[MAX_MODES];

    // vehicle set up
    VehicleParams params;
    VehicleState state;
    CostWeights weights;

    int costmap_size;
    size_t grid_size;
    std::vector<Gap> gaps;
    std::vector<Gaussian> modes;

    // transforms
    std::shared_ptr<tf2_ros::Buffer> tfBuffer;
    std::shared_ptr<tf2_ros::TransformListener> tfListener;

    // subscribers
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odomSub;
    rclcpp::Subscription<autoware_auto_planning_msgs::msg::Trajectory>::SharedPtr trajSub;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr mapSub;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scanSub;

    // publishers
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr controllerPub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vizPub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr modePub;

    // timer
    rclcpp::TimerBase::SharedPtr controlTimer;

    // messages
    nav_msgs::msg::Odometry::SharedPtr odom;
    autoware_auto_planning_msgs::msg::Trajectory::SharedPtr traj;
    nav_msgs::msg::OccupancyGrid::SharedPtr map;
    sensor_msgs::msg::LaserScan::SharedPtr scan;
};