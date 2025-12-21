#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <autoware_auto_planning_msgs/msg/trajectory.hpp>
#include <chrono>
#include <mppi/vehicle_util.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>

struct ControlInput
{
    double acceleration, steering; // integrate during publishing to command velocity and steering
};

struct CostWeights
{
    double qX, qY, qHeading;
    double qVx, qVy, qYawRate;
    double rAccel, rSteering;
};

class MPPI_Controller : public rclcpp::Node
{
  public:
    MPPI_Controller();

    void loadParameters();
    void updateState(const nav_msgs::msg::Odometry::SharedPtr odom);
    void stepDynamics();

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
    std::vector<ControlInput> nominalControlSequence;
    float sigmaAcceleration, sigmaSteering; // control noise for sampling

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