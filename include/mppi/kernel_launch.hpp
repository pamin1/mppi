#pragma once

#include <curand_kernel.h>
#include <mppi/vehicle_util.hpp>

void launchSetupRNG(curandState *d_states, unsigned long seed, int grid, int block);

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
 * @param grid Size of grid
 * @param block Size of block
 */
void launchMPPIKernel(ControlInput *d_controlSamples, double *d_costs, const ControlInput *d_nominalSequence, const VehicleState *d_refTraj, const VehicleState *d_currState, const CostWeights *d_weights, const VehicleParams *d_params, curandState *d_rngStates, int samples, int horizon, float dt, float sigmaAccel, float sigmaSteering, int grid, int block);