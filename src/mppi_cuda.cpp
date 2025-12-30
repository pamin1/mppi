#include <mppi/mppi_cuda.hpp>

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

    // distribution tuning
    this->declare_parameter("mppi.accel_dist", 2.0);
    this->declare_parameter("mppi.steer_dist", 0.3);

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

MPPI_Controller::~MPPI_Controller()
{
    RCLCPP_INFO(this->get_logger(), "Shutting down MPPI Controller...");

    cudaFree(d_nominalControls);
    cudaFree(d_refTraj);
    cudaFree(d_currState);
    cudaFree(d_weights);
    cudaFree(d_params);
    cudaFree(d_rngStates);
    cudaFree(d_controls);
    cudaFree(d_costs);

    RCLCPP_INFO(this->get_logger(), "CUDA memory freed");
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

    sigmaAcceleration = this->get_parameter("mppi.accel_dist").as_double();
    sigmaSteering = this->get_parameter("mppi.steer_dist").as_double();

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

    // set up GPU arrays
    cudaMalloc((void **)&d_nominalControls, sizeof(ControlInput) * horizon);
    cudaMalloc((void **)&d_refTraj, sizeof(VehicleState) * horizon);
    cudaMalloc((void **)&d_currState, sizeof(VehicleState));
    cudaMalloc((void **)&d_weights, sizeof(CostWeights));
    cudaMalloc((void **)&d_params, sizeof(VehicleParams));
    cudaMalloc((void **)&d_rngStates, sizeof(curandState) * samples);
    cudaMalloc((void **)&d_controls, sizeof(ControlInput) * samples * horizon); // 30 * 10000 * 16B = 3MB+??
    cudaMalloc((void **)&d_costs, sizeof(double) * samples);

    // copy fixed values over
    cudaMemcpy(d_weights, &weights, sizeof(CostWeights), cudaMemcpyHostToDevice);
    cudaMemcpy(d_params, &params, sizeof(VehicleParams), cudaMemcpyHostToDevice);

    block = 512;
    grid = (samples + block - 1) / block;

    // launch the rng kernel
    std::random_device rd;
    unsigned long seed = rd();

    launchSetupRNG(d_rngStates, seed, grid, block);
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
void MPPI_Controller::updateTraj(const autoware_auto_planning_msgs::msg::Trajectory::SharedPtr traj)
{
    trajectory.clear();
    for (auto &point : traj->points) // need to guarantee that we only take sample number of trajectory points
    {
        VehicleState currState;

        // vehicle state and trajectory should be in same map frame -- no tranform required
        currState.x = point.pose.position.x;
        currState.y = point.pose.position.y;

        tf2::Quaternion quat;
        tf2::fromMsg(point.pose.orientation, quat);
        tf2::Matrix3x3 m(quat);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        currState.heading = yaw;

        currState.vx = point.longitudinal_velocity_mps;
        currState.vy = 0.0; // assume no slipping from path planner
        currState.yawRate = point.heading_rate_rps;

        trajectory.push_back(currState);

        if (trajectory.size() == horizon)
        {
            return;
        }
    }
}

void MPPI_Controller::updateControl()
{
    // safety check for data available
    if (!odom || !traj)
    {
        RCLCPP_INFO(this->get_logger(), "Waiting for odometry or trajectory");
        return;
    }

    // update the state and strajectory
    try
    {
        updateState(odom);
        updateTraj(traj);
    }
    catch (...)
    {
        RCLCPP_ERROR(this->get_logger(), "Caught error attempting to update vehicle state");
    }

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

    // need to update the nominal control sequence, refTraj, and currState before iterating mppi
    cudaMemcpy(d_nominalControls, nominalControlSequence.data(), sizeof(ControlInput) * horizon, cudaMemcpyHostToDevice);
    cudaMemcpy(d_refTraj, trajectory.data(), sizeof(VehicleState) * horizon, cudaMemcpyHostToDevice);
    cudaMemcpy(d_currState, &state, sizeof(VehicleState), cudaMemcpyHostToDevice);

    launchMPPIKernel(d_controls, d_costs, d_nominalControls, d_refTraj, d_currState, d_weights, d_params, d_rngStates, samples, horizon, dt, sigmaAcceleration, sigmaSteering, grid, block);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MPPI_Controller>());
    rclcpp::shutdown();

    return 0;
}