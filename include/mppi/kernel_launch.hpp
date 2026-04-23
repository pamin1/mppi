#pragma once

#include <curand_kernel.h>
#include <mppi/mppi_util.hpp>

void launchSetupRNG(curandState *d_states, unsigned long seed, int grid, int block);

/**
 * @brief Creates the distribution of inputs for a single timestep. Timesteps are iterated on CPU, this computes the cost to go for the timestep on GPU
 * @param controlSamples Output array of sample control inputs
 * @param config Struct of the MPPI configuration
 * @param map Struct of Costmap information
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
 * @param grid Size of grid
 * @param block Size of block
 * @param stream cudaStream association for the kernel launch
 */
void launchMPPIKernel(ControlInput *d_controlSamples, MPPIConfig config, CostmapInfo *map, double *d_costs, const ControlInput *d_nominalSequence, const VehicleState *d_refTraj, const VehicleState *d_currState, const CostWeights *d_weights, const VehicleParams *d_params, curandState *d_rngStates, int grid, int block, cudaStream_t stream);

/**
 * @brief Executes the thrust device functions
 * @param d_costs Control input costs
 * @param samples Number of samples
 * @param temperature Distribution of exponential weighting for costs
 * @param stream cudaStream association for the kernel launch
 * @return minimum cost of control samples
 */
double launchThrustWeighting(double *d_costs, int samples, float temperature, cudaStream_t stream);

/**
 * @brief Computes the weigthed optimal control input for each time step in the horizon
 * @param optimalControls Optimal aggregate of the weighted control inputs
 * @param sampleControls Input sample control sequences
 * @param weightedCosts Array of costs for each sample control input
 * @param samples Number of path samples
 * @param horizon Length of horizon
 * @param grid Size of grid
 * @param block Size of block
 */
void launchAggregateControls(ControlInput *d_optimalControls, const ControlInput *d_sampleControls, const ControlInput *d_nominalControls, const double *d_weightedCosts, float alpha, int samples, int horizon, int grid, int block, cudaStream_t stream);