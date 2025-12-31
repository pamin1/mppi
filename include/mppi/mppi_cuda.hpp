#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <autoware_auto_planning_msgs/msg/trajectory.hpp>
#include <chrono>
#include <mppi/kernel_launch.hpp>
#include <mppi/vehicle_util.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <random>
#include <rclcpp/rclcpp.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

struct weightFunctor
{
    // member variables (state)
    double minCost;
    double lambda;

    // constructor to initialize state
    __host__ __device__ weightFunctor(double _min, double _lambda)
        : minCost(_min), lambda(_lambda)
    {
    }

    // function operator
    __host__ __device__ double operator()(double cost) const
    {
        return exp(-(cost - minCost) / lambda);
    }
};

class MPPI_Controller : public rclcpp::Node
{
  public:
    MPPI_Controller();
    ~MPPI_Controller();

    void loadParameters();
    void updateState(const nav_msgs::msg::Odometry::SharedPtr odom);
    void updateTraj(const autoware_auto_planning_msgs::msg::Trajectory::SharedPtr traj);

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void trajectoryCallback(const autoware_auto_planning_msgs::msg::Trajectory::SharedPtr msg);
    void updateControl();

  private:
    // controller set up
    std::string mapFrame = "map";
    std::string baseFrame = "ego_racecar/base_link";

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

    // GPU arrays
    int block, grid;

    ControlInput *d_nominalControls;
    VehicleState *d_refTraj;
    VehicleState *d_currState;

    CostWeights *d_weights;
    VehicleParams *d_params;

    curandState *d_rngStates;

    ControlInput *d_controls;
    double *d_costs;

    // vehicle set up
    VehicleParams params;
    VehicleState state;
    CostWeights weights;

    // transforms
    std::shared_ptr<tf2_ros::Buffer> tfBuffer;
    std::shared_ptr<tf2_ros::TransformListener> tfListener;

    // subscribers
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odomSub;
    rclcpp::Subscription<autoware_auto_planning_msgs::msg::Trajectory>::SharedPtr trajSub;

    // publishers
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr controllerPub;

    // timer
    rclcpp::TimerBase::SharedPtr controlTimer;

    // messages
    nav_msgs::msg::Odometry::SharedPtr odom;
    autoware_auto_planning_msgs::msg::Trajectory::SharedPtr traj;
};