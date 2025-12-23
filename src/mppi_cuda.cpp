#include <mppi/mppi_cuda.hpp>

// eigen for the matrix/vector math?

// make an include file for the vehicle dynamics and the function stepping

MPPI_Controller::MPPI_Controller()
    : rclcpp::Node("mppi_controller")
{
    // default controller params
    this->declare_parameter("mppi.samples", 10000);
    this->declare_parameter("mppi.control_frequency", 50);
    this->declare_parameter("mppi.horizon", 30);
    this->declare_parameter("mppi.temperature", 1.0);
    this->declare_parameter("mppi.alpha", 0.1);

    // state tracking weights
    this->declare_parameter("cost_weights.q_x", 1.0);
    this->declare_parameter("cost_weights.q_y", 1.0);
    this->declare_parameter("cost_weights.q_heading", 10.0);
    this->declare_parameter("cost_weights.q_vx", 1.0);
    this->declare_parameter("cost_weights.q_vy", 0.5);
    this->declare_parameter("cost_weights.q_yaw_rate", 0.5);

    // control effort weights
    this->declare_parameter("cost_weights.r_accel", 0.1);
    this->declare_parameter("cost_weights.r_steering", 0.1);
    loadParameters();

    // tranforms
    tfBuffer = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tfListener = std::make_shared<tf2_ros::TransformListener>(*tfBuffer);

    // subscribers
    odomSub = this->create_subscription<nav_msgs::msg::Odometry>("/ego_racecar/odom", 10, std::bind(&MPPI_Controller::odomCallback, this, std::placeholders::_1));
    trajSub = this->create_subscription<autoware_auto_planning_msgs::msg::Trajectory>("/planner_traj", 10, std::bind(&MPPI_Controller::trajectoryCallback, this, std::placeholders::_1));

    // publishers
    controllerPub = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);

    // timer
    controlTimer = this->create_wall_timer(std::chrono::duration<float>(dt), std::bind(&MPPI_Controller::updateControl, this));
    RCLCPP_INFO(this->get_logger(), "MPPI Controller initialized successfully");
}

void MPPI_Controller::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    odom = msg;
}

void MPPI_Controller::trajectoryCallback(const autoware_auto_planning_msgs::msg::Trajectory::SharedPtr msg)
{
    traj = msg;
}

void MPPI_Controller::loadParameters()
{
    samples = this->get_parameter("mppi.samples").as_int();
    controlFrequency = this->get_parameter("mppi.control_frequency").as_int();
    dt = 1.0f / controlFrequency;
    horizon = this->get_parameter("mppi.horizon").as_int();
    temperature = this->get_parameter("mppi.temperature").as_double();
    alpha = this->get_parameter("mppi.alpha").as_double();

    weights.qX = this->get_parameter("cost_weights.q_x").as_double();
    weights.qY = this->get_parameter("cost_weights.q_y").as_double();
    weights.qHeading = this->get_parameter("cost_weights.q_heading").as_double();
    weights.qVx = this->get_parameter("cost_weights.q_vx").as_double();
    weights.qVy = this->get_parameter("cost_weights.q_vy").as_double();
    weights.qYawRate = this->get_parameter("cost_weights.q_yaw_rate").as_double();
    weights.rAccel = this->get_parameter("cost_weights.r_accel").as_double();
    weights.rSteering = this->get_parameter("cost_weights.r_steering").as_double();

    RCLCPP_INFO(this->get_logger(), "MPPI Parameters:");
    RCLCPP_INFO(this->get_logger(), "\tSamples:\t%d", samples);
    RCLCPP_INFO(this->get_logger(), "\tFrequency:\t%d Hz", controlFrequency);
    RCLCPP_INFO(this->get_logger(), "\tHorizon:\t%d steps", horizon);
    RCLCPP_INFO(this->get_logger(), "\tTemperature:\t%.3f", temperature);
}

void MPPI_Controller::updateState(const nav_msgs::msg::Odometry::SharedPtr odom)
{
    geometry_msgs::msg::TransformStamped transform = tfBuffer->lookupTransform(mapFrame, baseFrame, rclcpp::Time());
    state.x = transform.transform.translation.x;
    state.y = transform.transform.translation.y;

    tf2::Quaternion quat(
        transform.transform.rotation.x,
        transform.transform.rotation.y,
        transform.transform.rotation.z,
        transform.transform.rotation.w);
    tf2::Matrix3x3 m(quat);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    state.heading = yaw;

    state.vx = odom->twist.twist.linear.x;
    state.vy = odom->twist.twist.linear.y;
    state.yawRate = odom->twist.twist.angular.z;
}

void MPPI_Controller::updateControl()
{
    // should perform another mppi implementation

    // get the initial control sequence -- warm starting
    if (!controlSeqInitialized)
    {
        nominalControlSequence.resize(horizon);
        for (auto &u : nominalControlSequence)
        {
            u.acceleration = 0.0;
            u.steering = 0.0;
        }
        controlSeqInitialized = true;
    }
    // sample the system dynamics (updateState?)
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MPPI_Controller>());
    rclcpp::shutdown();

    return 0;
}