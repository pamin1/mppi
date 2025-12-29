#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <mppi/vehicle_util.hpp>

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
__global__ void setupRNG(curandState *states, unsigned long seed)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &states[idx]);
}

/**
 * @brief Computes cost as error between current/predicted state and the reference state from optimized trajectory
 */
__device__ double computeCost(const VehicleState &predicted, const VehicleState &reference, const ControlInput &control, const CostWeights &weights)
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
 * @brief Creates the distribution of inputs for a single timestep. Timesteps are iterated on CPU, this computes the cost to go for the timestep on GPU
 * @param controlSamples Output array of sample control inputs
 * @param costs Output array of rollout costs
 * @param nominalControlSequence Input array of previous control sequence to build on
 * @param refTrajectory Input array of optimized path planning trajectory points
 * @param currState Input of the current vehicle state
 * @param weights Input of controller cost function weights
 * @param params Vehicle model parameters
 * @param states cuRAND states for RNG
 * @param samples Size of sample distribution
 * @param horizon Number of timesteps to take
 * @param sigmaAccel Standard Deviation for acceleration
 * @param sigmaSteering Standard Deviation for steering
 */
__global__ void mppiKernel(ControlInput *controlSamples, double *costs, const ControlInput *nominalControlSequence, const VehicleState *refTrajectory, VehicleState *currState, const CostWeights *weights, const VehicleParams *params, curandState *states, int samples, int horizon, float dt, float sigmaAccel, float sigmaSteering, float minAccel, float maxAccel, float minSteer, float maxSteer);