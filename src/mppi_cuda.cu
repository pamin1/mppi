#include <algorithm>
#include <mppi/mppi_cuda.hpp>
#include <numeric>

MPPI_Controller::MPPI_Controller()
    : rclcpp::Node("mppi_controller")
{
    loadParameters();

    // tranforms
    tfBuffer = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tfListener = std::make_shared<tf2_ros::TransformListener>(*tfBuffer);

    // subscribers
    odomSub = this->create_subscription<nav_msgs::msg::Odometry>("/ego_racecar/odom", 10, std::bind(&MPPI_Controller::odomCallback, this, std::placeholders::_1));
    trajSub = this->create_subscription<autoware_auto_planning_msgs::msg::Trajectory>("/planner_traj", 10, std::bind(&MPPI_Controller::trajectoryCallback, this, std::placeholders::_1));
    mapSub = this->create_subscription<nav_msgs::msg::OccupancyGrid>("/local_costmap", 10, std::bind(&MPPI_Controller::mapCallback, this, std::placeholders::_1));
    scanSub = this->create_subscription<sensor_msgs::msg::LaserScan>("/scan", 10, std::bind(&MPPI_Controller::scanCallback, this, std::placeholders::_1));

    // publishers
    controllerPub = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
    vizPub = this->create_publisher<visualization_msgs::msg::MarkerArray>("/mppi/top_k_paths", 10);
    modePub = this->create_publisher<visualization_msgs::msg::MarkerArray>("/mppi/steering_modes", 10);

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
    cudaFree(d_costmap_data);
    cudaFree(d_costmap_info);

    for (int i = 0; i < MAX_MODES; i++)
    {
        cudaFree(d_optimal_per_mode[i]);
        cudaStreamDestroy(streams[i]);
    }

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

void MPPI_Controller::mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
    map = msg;
}

void MPPI_Controller::scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
    scan = msg;
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

    // default controller params
    this->declare_parameter("mppi.samples", 10000);
    this->declare_parameter("mppi.control_frequency", 50);
    this->declare_parameter("mppi.horizon", 30);
    this->declare_parameter("mppi.temperature", 1.0);
    this->declare_parameter("mppi.alpha", 0.1);

    // distribution tuning
    this->declare_parameter("mppi.accel_dist", 2.0);
    this->declare_parameter("mppi.steer_dist", 0.3);
    this->declare_parameter("mppi.n_sigma", 3);

    // state tracking weights
    this->declare_parameter("cost_weights.q_x", 1.0);
    this->declare_parameter("cost_weights.q_y", 1.0);
    this->declare_parameter("cost_weights.q_heading", 10.0);
    this->declare_parameter("cost_weights.q_vx", 1.0);
    this->declare_parameter("cost_weights.q_vy", 0.5);
    this->declare_parameter("cost_weights.q_yaw_rate", 0.5);
    this->declare_parameter("cost_weights.r_accel", 0.1);
    this->declare_parameter("cost_weights.r_steering", 0.1);
    this->declare_parameter("cost_weights.r_steering_rate", 0.1);

    // map info
    this->declare_parameter("local_map.size", 10.0);
    this->declare_parameter("local_map.resolution", 0.05);

    // visualization
    this->declare_parameter("visualization.enable_viz", true);
    this->declare_parameter("visualization.top_k_paths", 10);
    this->declare_parameter("visualization.line_width", 0.02);
    this->declare_parameter("visualization.path_alpha", 0.7);

    samples = this->get_parameter("mppi.samples").as_int();
    controlFrequency = this->get_parameter("mppi.control_frequency").as_int();
    dt = 1.0f / controlFrequency;
    horizon = this->get_parameter("mppi.horizon").as_int();
    temperature = this->get_parameter("mppi.temperature").as_double();
    alpha = this->get_parameter("mppi.alpha").as_double();

    sigmaAcceleration = this->get_parameter("mppi.accel_dist").as_double();
    sigmaSteering = this->get_parameter("mppi.steer_dist").as_double();
    n_sigma = this->get_parameter("mppi.n_sigma").as_int();

    config.samples = samples;
    config.horizon = horizon;
    config.dt = dt;
    config.sigmaAcceleration = sigmaAcceleration;
    config.sigmaSteering = sigmaSteering;

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

    costmap_size = this->get_parameter("local_map.size").as_double();
    float resolution = this->get_parameter("local_map.resolution").as_double();
    int grid_width = static_cast<int>(costmap_size / resolution);
    grid_size = grid_width * grid_width;

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
    CHECK_ERROR(cudaMalloc((void **)&d_costmap_info, sizeof(CostmapInfo)));
    CHECK_ERROR(cudaMalloc((void **)&d_costmap_data, grid_size * sizeof(int8_t)));

    // copy fixed values over
    cudaMemcpy(d_weights, &weights, sizeof(CostWeights), cudaMemcpyHostToDevice);
    cudaMemcpy(d_params, &params, sizeof(VehicleParams), cudaMemcpyHostToDevice);

    for (int i = 0; i < MAX_MODES; i++)
    {
        cudaStreamCreate(&streams[i]);
        CHECK_ERROR(cudaMalloc(&d_optimal_per_mode[i], sizeof(ControlInput) * horizon));
    }

    block = 512;
    grid = (samples + block - 1) / block;

    // launch the rng kernel
    std::random_device rd;
    unsigned long seed = rd();

    launchSetupRNG(d_rngStates, seed, grid, block);
}

void MPPI_Controller::updateState(const nav_msgs::msg::Odometry::SharedPtr odom)
{
    double odom_age_ms = (this->now() - odom->header.stamp).seconds() * 1000.0;
    if (odom_age_ms > 2.0 * dt * 1000.0)
    {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "Stale odom: %.1f ms old", odom_age_ms);
    }

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
    double traj_age_ms = (this->now() - traj->header.stamp).seconds() * 1000.0;
    if (traj_age_ms > 2.0 * dt * 1000.0)
    {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "Stale trajectory: %.1f ms old", traj_age_ms);
    }

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

void MPPI_Controller::updateMap(const nav_msgs::msg::OccupancyGrid::SharedPtr map)
{
    double map_age_ms = (this->now() - map->header.stamp).seconds() * 1000.0;
    if (map_age_ms > 2.0 * dt * 1000.0)
    {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "Stale costmap: %.1f ms old", map_age_ms);
    }

    size_t incoming_size = map->info.width * map->info.height;

    // copy grid data to device
    cudaMemcpy(d_costmap_data, map->data.data(), grid_size * sizeof(int8_t), cudaMemcpyHostToDevice);

    // build struct with device pointer
    CostmapInfo costmap_info;
    costmap_info.data = d_costmap_data; // device pointer, allocated once in constructor
    costmap_info.height = map->info.height;
    costmap_info.width = map->info.width;
    costmap_info.resolution = map->info.resolution;
    costmap_info.origin_x = map->info.origin.position.x;
    costmap_info.origin_y = map->info.origin.position.y;
    costmap_info.lethal_cost = 100;

    // copy struct to device
    cudaMemcpy(d_costmap_info, &costmap_info, sizeof(CostmapInfo), cudaMemcpyHostToDevice);
}

void MPPI_Controller::updateScan(const sensor_msgs::msg::LaserScan::SharedPtr scan)
{
    double scan_age_ms = (this->now() - scan->header.stamp).seconds() * 1000.0;
    if (scan_age_ms > 2.0 * dt * 1000.0)
    {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "Stale scan: %.1f ms old", scan_age_ms);
    }

    int near_zero_count = 0;
    for (const auto& r : scan->ranges) {
        if (r < 0.05f) near_zero_count++;
    }
    if (near_zero_count > static_cast<int>(scan->ranges.size()) / 2) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 500, "Degenerate scan — skipping gap update");
        return;  // keep previous gaps/modes
    }

    // laserscan gap finding
    float r_min = 3.0;
    gaps = findGaps(scan, r_min);
    int num_gaps = gaps.size();
    // RCLCPP_INFO(this->get_logger(), "Num Gaps: %d", static_cast<int>(gaps.size()));

    // Build ranges: 0 outside gaps, original range inside gaps
    std::vector<float> gap_ranges(scan->ranges.size(), 0.0f);

    modes.clear();
    modes.reserve(num_gaps);

    for (int i = 0; i < gaps.size(); i++)
    {
        for (int j = gaps[i].start; j <= gaps[i].end; j++)
        {
            gap_ranges[j] = scan->ranges[j];
        }

        // gap finding
        int mid_idx = (gaps[i].start + gaps[i].end) / 2;
        double angle = scan->angle_min + mid_idx * scan->angle_increment;

        // angle = static_cast<float>(std::clamp(angle, -0.01, 0.01));
        int gap_width = gaps[i].end - gaps[i].start;
        float angular_width = gap_width * scan->angle_increment; // radians
        
        // Skip degenerate gaps
        if (gap_width < 5 || angular_width < 0.05f) continue;

        float std_dev = angular_width / (2 * n_sigma);
        std_dev = std::max(std_dev, 0.01f);  // hard floor

        Gaussian mode;
        mode.mean = angle;
        mode.std_dev = std_dev; // can modify later
        modes.push_back(mode);

        // RCLCPP_INFO(this->get_logger(), "Gap %d: mean: %f, std_dev: %f", i, angle, std_dev);
    }
}

void MPPI_Controller::updateControl()
{
    // begin timing
    auto start = std::chrono::high_resolution_clock::now();

    // safety check for data available
    if (!odom || !traj || !map || !scan)
    {
        RCLCPP_INFO(this->get_logger(), "Waiting for odometry or trajectory or map or scan");
        return;
    }

    // update the state and strajectory and costmap information
    try
    {
        updateState(odom);
        updateTraj(traj);
        updateMap(map);
        updateScan(scan);
    }
    catch (...)
    {
        RCLCPP_ERROR(this->get_logger(), "Caught error attempting to update from callback information");
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

    int agg_block = 32;
    int agg_grid = (horizon + agg_block - 1) / agg_block;
    
    int num_modes = std::min(static_cast<int>(modes.size()), MAX_MODES);
    
    if (num_modes == 0)
    {
        // Fallback: single unimodal rollout centered on zero steering
        MPPIConfig fallback_config = config;
        fallback_config.samples = samples;
        fallback_config.sigmaSteering = sigmaSteering;
        fallback_config.steeringBias = 0.0;

        int fallback_grid = (samples + block - 1) / block;

        launchMPPIKernel(d_controls, fallback_config, d_costmap_info, d_costs, d_nominalControls, d_refTraj, d_currState, d_weights, d_params, d_rngStates, fallback_grid, block, streams[0]);

        launchThrustWeighting(d_costs, samples, temperature, streams[0]);

        launchAggregateControls(d_optimal_per_mode[0], d_controls, d_nominalControls, d_costs, alpha, samples, horizon, agg_grid, agg_block, streams[0]);

        cudaStreamSynchronize(streams[0]);

        ControlInput control;
        cudaMemcpy(&control, d_optimal_per_mode[0], sizeof(ControlInput), cudaMemcpyDeviceToHost);

        float speed = state.vx + control.acceleration * dt;

        ackermann_msgs::msg::AckermannDriveStamped msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = baseFrame;
        msg.drive.speed = std::max(-params.maxVelocity, std::min(speed, params.maxVelocity));
        msg.drive.steering_angle = control.steering;
        controllerPub->publish(msg);

        // Warm-start: shift nominal sequence left by 1
        cudaMemcpy(d_nominalControls, d_optimal_per_mode[0] + 1, sizeof(ControlInput) * (horizon - 1), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_nominalControls + (horizon - 1), d_optimal_per_mode[0] + (horizon - 1), sizeof(ControlInput), cudaMemcpyDeviceToDevice);

        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 500, "No gaps — unimodal fallback");
        return;
    }

    int samples_per_mode = samples / num_modes;
    double mode_min_costs[MAX_MODES];

    for (int m = 0; m < num_modes; m++)
    {
        int offset = m * samples_per_mode;

        MPPIConfig mode_config = config; // copy base config
        mode_config.samples = samples_per_mode;
        mode_config.sigmaSteering = modes[m].std_dev;
        mode_config.steeringBias = modes[m].mean;
        // RCLCPP_INFO(this->get_logger(), "Angle: %.4f, Std Dev: %.4f", modes[m].mean, modes[m].std_dev);

        int mode_grid = (samples_per_mode + block - 1) / block;

        launchMPPIKernel(d_controls + offset * horizon, mode_config, d_costmap_info, d_costs + offset, d_nominalControls, d_refTraj, d_currState, d_weights, d_params, d_rngStates + offset, mode_grid, block, streams[m]);

        mode_min_costs[m] = launchThrustWeighting(d_costs + offset, samples_per_mode, temperature, streams[m]);

        launchAggregateControls(d_optimal_per_mode[m], d_controls + offset * horizon, d_nominalControls, d_costs + offset, alpha, samples_per_mode, horizon, agg_grid, agg_block, streams[m]);
    }

    for (int m = 0; m < num_modes; m++)
    {
        cudaStreamSynchronize(streams[m]);
    }

    int best = 0;
    for (int m = 1; m < num_modes; m++)
    {
        if (mode_min_costs[m] < mode_min_costs[best])
            best = m;
    }

    ControlInput control;
    cudaMemcpy(&control, d_optimal_per_mode[best], sizeof(ControlInput), cudaMemcpyDeviceToHost);

    float speed = state.vx + control.acceleration * dt;

    // publish the control
    ackermann_msgs::msg::AckermannDriveStamped msg;
    msg.header.stamp = this->now();
    msg.header.frame_id = baseFrame;
    msg.drive.speed = std::max(-params.maxVelocity, std::min(speed, params.maxVelocity));
    msg.drive.steering_angle = control.steering;
    controllerPub->publish(msg);

    // copy the optimal controls back to the nominal control input
    // shift left by 1, repeat last entry
    cudaMemcpy(d_nominalControls, d_optimal_per_mode[best] + 1, sizeof(ControlInput) * (horizon - 1), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_nominalControls + (horizon - 1), d_optimal_per_mode[best] + (horizon - 1), sizeof(ControlInput), cudaMemcpyDeviceToDevice);
    
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    // RCLCPP_INFO(this->get_logger(), "MPPI solve: %.2f ms (%.1f Hz capable)", ms, 1000.0 / ms);

    publishBestPaths(best, num_modes, mode_min_costs);
    publishModes(modes, scan->header.stamp, best);
}

void MPPI_Controller::publishBestPaths(int best_mode, int num_modes, double *mode_min_costs)
{
    visualization_msgs::msg::MarkerArray markerArray;

    visualization_msgs::msg::Marker deleteAll;
    deleteAll.action = visualization_msgs::msg::Marker::DELETEALL;
    markerArray.markers.push_back(deleteAll);

    // find worst mode
    int worst_mode = 0;
    for (int m = 1; m < num_modes; m++)
    {
        if (mode_min_costs[m] > mode_min_costs[worst_mode])
            worst_mode = m;
    }

    // copy both control sequences to host
    std::vector<ControlInput> bestControls(horizon);
    std::vector<ControlInput> worstControls(horizon);
    cudaMemcpy(bestControls.data(), d_optimal_per_mode[best_mode],
               sizeof(ControlInput) * horizon, cudaMemcpyDeviceToHost);
    cudaMemcpy(worstControls.data(), d_optimal_per_mode[worst_mode],
               sizeof(ControlInput) * horizon, cudaMemcpyDeviceToHost);

    struct PathViz
    {
        std::string ns;
        int id_line;
        int id_label;
        std::vector<ControlInput> &controls;
        float r, g, b;
        double cost;
        std::string label_prefix;
    };

    std::vector<PathViz> paths = {
        {"mppi_best_path", 0, 1, bestControls, 0.0f, 1.0f, 0.0f,
         mode_min_costs[best_mode], "BEST (mode " + std::to_string(best_mode) + ")"},
        {"mppi_worst_path", 2, 3, worstControls, 1.0f, 0.0f, 0.0f,
         mode_min_costs[worst_mode], "WORST (mode " + std::to_string(worst_mode) + ")"},
    };

    for (auto &p : paths)
    {
        // line strip
        visualization_msgs::msg::Marker line;
        line.header.stamp = this->now();
        line.header.frame_id = mapFrame;
        line.ns = p.ns;
        line.id = p.id_line;
        line.type = visualization_msgs::msg::Marker::LINE_STRIP;
        line.action = visualization_msgs::msg::Marker::ADD;
        line.scale.x = vizLineWidth;
        line.pose.orientation.w = 1.0;
        line.color.r = p.r;
        line.color.g = p.g;
        line.color.b = p.b;
        line.color.a = vizPathAlpha;

        VehicleState simState = state;
        geometry_msgs::msg::Point pt;
        pt.x = simState.x;
        pt.y = simState.y;
        pt.z = 0.0;
        line.points.push_back(pt);

        for (int t = 0; t < horizon; ++t)
        {
            simState = stepDynamics(simState, params, p.controls[t], dt);
            pt.x = simState.x;
            pt.y = simState.y;
            pt.z = 0.0;
            line.points.push_back(pt);
        }

        markerArray.markers.push_back(line);

        // cost label at the end of the path
        visualization_msgs::msg::Marker label;
        label.header.stamp = this->now();
        label.header.frame_id = mapFrame;
        label.ns = p.ns;
        label.id = p.id_label;
        label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        label.action = visualization_msgs::msg::Marker::ADD;
        label.pose.position.x = simState.x;
        label.pose.position.y = simState.y;
        label.pose.position.z = 0.5;
        label.pose.orientation.w = 1.0;
        label.scale.z = 0.3;
        label.color.r = p.r;
        label.color.g = p.g;
        label.color.b = p.b;
        label.color.a = 1.0f;

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << "cost: " << p.cost;
        label.text = oss.str();

        markerArray.markers.push_back(label);
    }

    vizPub->publish(markerArray);
}

void MPPI_Controller::publishModes(const std::vector<Gaussian> &modes, const rclcpp::Time &stamp, int best_mode)
{
    visualization_msgs::msg::MarkerArray marker_array;
    std::string costmap_frame_ = "ego_racecar/laser_model";

    // Clear previous markers
    visualization_msgs::msg::Marker clear;
    clear.header.stamp = stamp;
    clear.header.frame_id = costmap_frame_;
    clear.ns = "steering_modes";
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(clear);

    if (gaps.empty())
    {
        modePub->publish(marker_array);
        return;
    }

    // Compute mode endpoints and scores
    struct Point
    {
        double x;
        double y;
        double score;
    };
    std::vector<Point> points;
    points.reserve(gaps.size());

    for (const auto &mode : modes)
    {
        float angle = mode.mean;
        float range = 2.0;

        points.push_back({
            static_cast<double>(range * std::cos(angle)),
            static_cast<double>(range * std::sin(angle)),
            static_cast<double>(mode.std_dev) // width as score
        });
    }

    for (size_t i = 0; i < modes.size(); ++i)
    {
        visualization_msgs::msg::Marker m;
        m.header.stamp = stamp;
        m.header.frame_id = costmap_frame_;
        m.ns = "steering_modes";
        m.id = static_cast<int>(i);
        m.type = visualization_msgs::msg::Marker::ARROW;
        m.action = visualization_msgs::msg::Marker::ADD;

        geometry_msgs::msg::Point start;
        start.x = 0.0;
        start.y = 0.0;
        start.z = 0.1;
        geometry_msgs::msg::Point end;
        end.x = points[i].x;
        end.y = points[i].y;
        end.z = 0.1;
        m.points.push_back(start);
        m.points.push_back(end);

        m.scale.x = m.scale.y = m.scale.z = points[i].score;

        if (static_cast<int>(i) == best_mode)
        {
            m.color.r = 0.0;
            m.color.g = 1.0;
            m.color.b = 0.0;
        }
        else
        {
            m.color.r = 1.0;
            m.color.g = 0.0;
            m.color.b = 0.0;
        }
        m.color.a = 1.0;

        m.lifetime.sec = 0;
        m.lifetime.nanosec = 100000000; // 0.1s

        marker_array.markers.push_back(m);
    }

    modePub->publish(marker_array);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MPPI_Controller>());
    rclcpp::shutdown();

    return 0;
}