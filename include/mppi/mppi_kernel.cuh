#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <mppi/mppi_util.hpp>

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
    __device__ double operator()(double cost) const
    {
        return exp(-(cost - minCost) / lambda);
    }
};

__device__ inline float clamp(float value, float min_val, float max_val)
{
    return fminf(fmaxf(value, min_val), max_val);
}

__device__ inline double angleDiff(double a, double b)
{
    double diff = a - b;
    while (diff > M_PI)
        diff -= 2.0 * M_PI;
    while (diff < -M_PI)
        diff += 2.0 * M_PI;
    return diff;
}

/**
 * @brief Initializes sample number of states with RNG seeding for cuRand
 * @param states Storage of device RNG seeds
 * @param seed Device seed
 */
__global__ void setupRNG(curandState *states, unsigned long seed);

/**
 * @brief Computes cost as error between current/predicted state and the reference state from optimized trajectory
 */
__device__ __forceinline__ double computeCost(const VehicleState &predicted, const VehicleState &reference, const ControlInput &control, const CostWeights &weights)
{
    double cost = 0.0;

    // conpute errors
    cost += (predicted.x - reference.x) * (predicted.x - reference.x) * weights.qX;
    cost += (predicted.y - reference.y) * (predicted.y - reference.y) * weights.qY;
    cost += angleDiff(predicted.heading, reference.heading) * angleDiff(predicted.heading, reference.heading) * weights.qHeading;
    cost += (predicted.vx - reference.vx) * (predicted.vx - reference.vx) * weights.qVx;
    cost += (predicted.vy - reference.vy) * (predicted.vy - reference.vy) * weights.qVy;
    cost += (predicted.yawRate - reference.yawRate) * (predicted.yawRate - reference.yawRate) * weights.qYawRate;

    cost += weights.rAccel * control.acceleration * control.acceleration + weights.rSteering * control.steering * control.steering;

    return cost;
}

/**
 * @brief Checks for a collision against the Occupancy Grid map
 */
__device__ bool checkCollision(const CostmapInfo &map, int x, int y)
{
    int gx = __float2int_rd((x + map.origin_offset) / map.resolution);
    int gy = __float2int_rd((y + map.origin_offset) / map.resolution);

    if (gx >= 0 && gx < map.width && gy >= 0 && gy < map.height)
    {
        return map.data[gy * map.width + gx] >= 128;
    }
    return true; // out of bounds = collision
}

/**
 * @brief Creates the distribution of inputs for a single timestep. Timesteps are iterated on CPU, this computes the cost to go for the timestep on GPU
 * @param controlSamples Output array of sample control inputs
 * @param config Struct of the MPPI configuration values
 * @param costs Output array of rollout costs
 * @param nominalControlSequence Input array of previous control sequence to build on
 * @param refTrajectory Input array of optimized path planning trajectory points
 * @param currState Input of the current vehicle state
 * @param weights Input of controller cost function weights
 * @param params Vehicle model parameters
 * @param states cuRAND states for RNG
 */
__global__ void mppiKernel(ControlInput *controlSamples, MPPIConfig *config, double *costs, const ControlInput *nominalControlSequence, const VehicleState *refTrajectory, const VehicleState *currState, const CostWeights *weights, const VehicleParams *params, curandState *states);

/**
 * @brief Computes the weigthed optimal control input for each time step in the horizon
 * @param optimalControls Optimal aggregate of the weighted control inputs
 * @param sampleControls Input sample control sequences
 * @param weightedCosts Array of costs for each sample control input
 * @param samples Number of path samples
 * @param horizon Length of horizon
 */
__global__ void aggregateControls(ControlInput *optimalControls, const ControlInput *sampleControls, const ControlInput *nominalControls, const double *weightedCosts, float alpha, int samples, int horizon);