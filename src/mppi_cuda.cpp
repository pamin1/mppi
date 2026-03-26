#include <algorithm>
#include <mppi/mppi_cuda.hpp>
#include <numeric>

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
    this->declare_parameter("cost_weights.r_steering_rate", 0.1);

    // visualization
    this->declare_parameter("visualization.enable_viz", true);
    this->declare_parameter("visualization.top_k_paths", 10);
    this->declare_parameter("visualization.line_width", 0.02);
    this->declare_parameter("visualization.path_alpha", 0.7);
    loadParameters();

    // tranforms
    tfBuffer = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tfListener = std::make_shared<tf2_ros::TransformListener>(*tfBuffer);

    // subscribers
    odomSub = this->create_subscription<nav_msgs::msg::Odometry>("/ego_racecar/odom", 10, std::bind(&MPPI_Controller::odomCallback, this, std::placeholders::_1));
    trajSub = this->create_subscription<autoware_auto_planning_msgs::msg::Trajectory>("/planner_traj", 10, std::bind(&MPPI_Controller::trajectoryCallback, this, std::placeholders::_1));

    // publishers
    controllerPub = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
    vizPub = this->create_publisher<visualization_msgs::msg::MarkerArray>("/mppi/top_k_paths", 10);

    // timer
    controlTimer = this->create_wall_timer(std::chrono::duration<float>(dt), std::bind(&MPPI_Controller::updateControl, this));
    RCLCPP_INFO(this->get_logger(), "MPPI Controller initialized successfully");
}

MPPI_Controller::~MPPI_Controller()
{
    RCLCPP_INFO(this->get_logger(), "Shutting down MPPI Controller...");

    cudaFree(d_optimalControls);
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
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        RCLCPP_INFO(this->get_logger(), "error: invalid device choosen\n");
        return;
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "Device: %d", deviceCount);
    }

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
    weights.rSteeringRate = this->get_parameter("cost_weights.r_steering_rate").as_double();

    enableViz = this->get_parameter("visualization.enable_viz").as_bool();
    topKPaths = this->get_parameter("visualization.top_k_paths").as_int();
    vizLineWidth = static_cast<float>(this->get_parameter("visualization.line_width").as_double());
    vizPathAlpha = static_cast<float>(this->get_parameter("visualization.path_alpha").as_double());

    RCLCPP_INFO(this->get_logger(), "Loaded parameters:");
    RCLCPP_INFO(this->get_logger(), "  samples: %d", samples);
    RCLCPP_INFO(this->get_logger(), "  control_frequency: %d", controlFrequency);
    RCLCPP_INFO(this->get_logger(), "  dt: %.4f", dt);
    RCLCPP_INFO(this->get_logger(), "  horizon: %d", horizon);
    RCLCPP_INFO(this->get_logger(), "  temperature: %.3f", temperature);
    RCLCPP_INFO(this->get_logger(), "  alpha: %.3f", alpha);
    RCLCPP_INFO(this->get_logger(), "  sigma_accel: %.3f", sigmaAcceleration);
    RCLCPP_INFO(this->get_logger(), "  sigma_steering: %.3f", sigmaSteering);
    RCLCPP_INFO(this->get_logger(), "Cost weights:");
    RCLCPP_INFO(this->get_logger(), "  q_x: %.3f", weights.qX);
    RCLCPP_INFO(this->get_logger(), "  q_y: %.3f", weights.qY);
    RCLCPP_INFO(this->get_logger(), "  q_heading: %.3f", weights.qHeading);
    RCLCPP_INFO(this->get_logger(), "  q_vx: %.3f", weights.qVx);
    RCLCPP_INFO(this->get_logger(), "  q_vy: %.3f", weights.qVy);
    RCLCPP_INFO(this->get_logger(), "  q_yaw_rate: %.3f", weights.qYawRate);
    RCLCPP_INFO(this->get_logger(), "  r_accel: %.3f", weights.rAccel);
    RCLCPP_INFO(this->get_logger(), "  r_steering: %.3f", weights.rSteering);
    RCLCPP_INFO(this->get_logger(), "  r_steering_rate: %.3f", weights.rSteeringRate);

    // set up GPU arrays
    CHECK_ERROR(cudaMalloc((void **)&d_optimalControls, sizeof(ControlInput) * horizon));
    CHECK_ERROR(cudaMalloc((void **)&d_nominalControls, sizeof(ControlInput) * horizon));
    CHECK_ERROR(cudaMalloc((void **)&d_refTraj, sizeof(VehicleState) * horizon));
    CHECK_ERROR(cudaMalloc((void **)&d_currState, sizeof(VehicleState)));
    CHECK_ERROR(cudaMalloc((void **)&d_weights, sizeof(CostWeights)));
    CHECK_ERROR(cudaMalloc((void **)&d_params, sizeof(VehicleParams)));
    CHECK_ERROR(cudaMalloc((void **)&d_rngStates, sizeof(curandState) * samples));
    CHECK_ERROR(cudaMalloc((void **)&d_controls, sizeof(ControlInput) * samples * horizon)); // 30 * 10000 * 16B = 3MB+??
    CHECK_ERROR(cudaMalloc((void **)&d_costs, sizeof(double) * samples));

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
        return;
    }

    // get the initial control sequence -- warm starting
    if (!controlSeqInitialized)
    {
        nominalControlSequence.resize(horizon);
        for (auto &u : nominalControlSequence)
        {
            u.acceleration = 0.1;
            u.steering = 0.0;
        }
        cudaMemcpy(d_nominalControls, nominalControlSequence.data(), sizeof(ControlInput) * horizon, cudaMemcpyHostToDevice);
        controlSeqInitialized = true;
    }

    // need to update the nominal control sequence, refTraj, and currState before iterating mppi
    cudaMemcpy(d_refTraj, trajectory.data(), sizeof(VehicleState) * horizon, cudaMemcpyHostToDevice);
    cudaMemcpy(d_currState, &state, sizeof(VehicleState), cudaMemcpyHostToDevice);

    // cost function is the error
    launchMPPIKernel(d_controls, d_costs, d_nominalControls, d_refTraj, d_currState, d_weights, d_params, d_rngStates, samples, horizon, dt, sigmaAcceleration, sigmaSteering, grid, block);

    launchThrustWeighting(d_costs, samples, temperature);

    if (enableViz)
    {
        std::vector<double> weights(samples);
        std::vector<ControlInput> allControls(samples * horizon);
        cudaMemcpy(weights.data(), d_costs, sizeof(double) * samples, cudaMemcpyDeviceToHost);
        cudaMemcpy(allControls.data(), d_controls, sizeof(ControlInput) * samples * horizon, cudaMemcpyDeviceToHost);
        publishTopKPaths(weights, allControls);
    }

    int agg_block = 32;
    int agg_grid = (horizon + agg_block - 1) / agg_block;
    launchAggregateControls(d_optimalControls, d_controls, d_nominalControls, d_costs, alpha, samples, horizon, agg_grid, agg_block);

    // d_optimalControls now has a single optimal control sequence + blended with nominal control for smoothness
    ControlInput control;
    cudaMemcpy(&control, d_optimalControls, sizeof(ControlInput), cudaMemcpyDeviceToHost);

    // publish the control
    ackermann_msgs::msg::AckermannDriveStamped msg;
    msg.header.stamp = this->now();
    msg.header.frame_id = baseFrame;
    msg.drive.speed = std::clamp(static_cast<float>(state.vx + control.acceleration * dt), 
                                  -params.maxVelocity, params.maxVelocity);
    msg.drive.steering_angle = control.steering;
    controllerPub->publish(msg);

    // copy the optimal controls back to the nominal control input
    // shift left by 1, repeat last entry
    cudaMemcpy(d_nominalControls, d_optimalControls + 1, sizeof(ControlInput) * (horizon - 1), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_nominalControls + (horizon - 1), d_optimalControls + (horizon - 1), sizeof(ControlInput), cudaMemcpyDeviceToDevice);
}

void MPPI_Controller::publishTopKPaths(const std::vector<double> &weights, const std::vector<ControlInput> &allControls)
{
    int k = std::min(topKPaths, samples);

    // Find top-k sample indices by highest weight (higher weight = lower cost = better path)
    std::vector<int> indices(samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&weights](int a, int b)
                      { return weights[a] > weights[b]; });

    visualization_msgs::msg::MarkerArray markerArray;

    // Delete stale markers from previous publish (in case k shrank)
    visualization_msgs::msg::Marker deleteAll;
    deleteAll.action = visualization_msgs::msg::Marker::DELETEALL;
    markerArray.markers.push_back(deleteAll);

    for (int rank = 0; rank < k; ++rank)
    {
        int sampleIdx = indices[rank];

        visualization_msgs::msg::Marker marker;
        marker.header.stamp = this->now();
        marker.header.frame_id = mapFrame;
        marker.ns = "mppi_paths";
        marker.id = rank;
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = vizLineWidth;
        marker.pose.orientation.w = 1.0;

        // Green (rank 0, best) -> Red (rank k-1, worst of top-k)
        float t = (k > 1) ? static_cast<float>(rank) / static_cast<float>(k - 1) : 0.0f;
        marker.color.r = t;
        marker.color.g = 1.0f - t;
        marker.color.b = 0.0f;
        marker.color.a = vizPathAlpha;

        // Simulate trajectory forward from current state using this sample's controls
        VehicleState simState = state;

        geometry_msgs::msg::Point pt;
        pt.x = simState.x;
        pt.y = simState.y;
        pt.z = 0.0;
        marker.points.push_back(pt);

        for (int t_step = 0; t_step < horizon; ++t_step)
        {
            const ControlInput &ctrl = allControls[sampleIdx * horizon + t_step];
            simState = stepDynamics(simState, params, ctrl, dt);
            pt.x = simState.x;
            pt.y = simState.y;
            pt.z = 0.0;
            marker.points.push_back(pt);
        }

        markerArray.markers.push_back(marker);
    }

    vizPub->publish(markerArray);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MPPI_Controller>());
    rclcpp::shutdown();

    return 0;
}